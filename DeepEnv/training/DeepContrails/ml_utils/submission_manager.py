

import pandas as pd
import numpy as np
from typing import List, Union, Callable
import dill
import torch
import gc
import os
import joblib

import matplotlib.pyplot as plt
from joblib import Parallel, delayed    
from itertools import combinations


from DeepContrail.image_utils.image_processing import load_image

def get_combinations(lst, n):
    return list(combinations(lst, n))

def flatten_list(lst):
    result = []
    for i in lst:
        if isinstance(i, list):
            result.extend(flatten_list(i))
        else:
            result.append(i)
    return result

def get_num_denum_img(y_pred, y_true, xs):

    
    scores = []
    for x in xs:
        y_p = np.where(y_pred > x, 1.0, 0.0).flatten()
        y_t = y_true.flatten()
        intersection = np.sum(y_p * y_t)
        union = np.sum(y_p) + np.sum(y_t)
        scores.append(np.array([intersection, union]))
    return scores



class SubmissionManager:
    def __init__(self, model_names: List[str], 
                 data_path: str, 
                 model_path: str, 
                 context = False,
                 meta_model_name: str = None,
                 batch_size: int = 16,
                 metric: Union[str, Callable] = None,
                 tta : bool = False
                ):
        """
        Initializes the SubmissionManager class.
        
        Parameters:
            model_names (List[str]): List of model names.
            data_path (str): Path to the data.
            model_path (str): Path to the models.
            batch_size (int, optional): Batch size for model inference. Default is 16.
        """
        self.model_names = model_names
        self.model_path = model_path
        self.data_path = data_path
        self.batch_size = batch_size
        self.metric = metric
        self.context = context
        self.meta_model_name = meta_model_name
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tta = tta
        
    def model_submission(self, model_name: str, 
                         return_dict=False, 
                         case="test",
                         threshold=True,
                         proba=True,
                         modulo=1,
                         keep=0,
                         xy_dict=None, 
                         fold=None,
                         calibrated=False,
                         video=False) -> pd.DataFrame:
        """
        Retrieves the submission dataframe for a specific model.

        Parameters:
            model_name (str): Name of the model.

        Returns:
            pd.DataFrame: Submission dataframe.
        """
        if case == "test" or case == "validation":
            instance_path = self.model_path + model_name + "/" + model_name + "/instance.pkl"
            model_path = self.model_path + model_name + "/" + model_name + "/model.ckpt"
        else:
            instance_path = self.model_path + model_name + "/instance.pkl"
            model_path = self.model_path + model_name + "/model.ckpt"

        with open(instance_path, 'rb') as file:
            trainer = dill.load(file)
            trainer.load_model(path=model_path)
        if not hasattr(trainer, 'logits'):
            trainer.logits = True
        if not hasattr(trainer, 'panel_image'):
            trainer.panel_image = False
            trainer.model.panel_image = False
        if not hasattr(trainer, 'base_resize'):
            trainer.base_resize = 256
            trainer.model.base_resize = 256
        if trainer.data_augmentation :
            import ttach as tta
            trainer.model.model = tta.SegmentationTTAWrapper(trainer.model.model, tta.aliases.d4_transform(), merge_mode='mean')
        
        result = trainer.submit_result(self.data_path,
                                       batch_size=self.batch_size, 
                                       return_dict=return_dict,
                                       threshold=threshold,
                                       case=case,
                                       proba=proba,
                                       modulo=modulo,
                                       keep=keep,
                                       xy_dict=xy_dict,
                                       fold=fold,
                                       calibrated=calibrated,
                                       video=video)
        
        trainer.model.cpu()
        trainer.model = None
        del trainer.model
        trainer = None
        del trainer
        torch.cuda.empty_cache()
        return result

    def submit_result(self, 
                      threshold=True, 
                      case="test",
                      split=3,
                      calibrated=False,
                      
                      proba=True) -> pd.DataFrame:
        """
        Submits the final results by combining predictions from multiple models.
        
        Parameters:
            threshold (float, optional): Threshold for binary classification. Default is 0.5.
        
        Returns:
            pd.DataFrame: Final submission dataframe.
        """
        if len(self.model_names) == 1:
            return self.model_submission(self.model_names[0], case=case)
        
        else:
            
            
            try:
                submission = pd.read_csv(self.data_path + '/sample_submission.csv', index_col='record_id')
            except FileNotFoundError:
                submission = pd.DataFrame(columns=["record_id","encoded_pixels"])
                submission.record_id = [int(el) for el in os.listdir(self.data_path + case) if ".json" not in el]
                submission = submission.set_index("record_id")

            gc.enable()
            if self.meta_model_name is None:
                submission_dict = {} 
                for model_name in self.model_names:
                    model_dict = self.model_submission(model_name, 
                                                            return_dict="torch", 
                                                            threshold=False if calibrated else True,
                                                            proba=False if calibrated else True,
                                                            calibrated=calibrated,
                                                            case=case,
                                                            )
                    
                    for idx, value in model_dict.items():

                        try:
                            submission_dict[idx] += value.squeeze(0)
                        except KeyError:
                            submission_dict[idx] = torch.zeros((256,256)) + value.squeeze(0)
                    
                    gc.collect()
                

                for idx in submission.index:
                    predicted_mask = submission_dict[idx] / len(self.model_names)
                    if calibrated:
                        predicted_mask = torch.sigmoid(predicted_mask)
                    predicted_mask= predicted_mask.cpu().detach().numpy()

                    predicted_mask_with_threshold = np.zeros((256, 256))
                    predicted_mask_with_threshold[predicted_mask <= threshold] = 0
                    predicted_mask_with_threshold[predicted_mask > threshold] = 1

                    submission.loc[int(idx), 'encoded_pixels'] = list_to_string(rle_encode(predicted_mask_with_threshold))
            else:
                for keep in range(split):
                    xy_dict = {} 

                    for model_name in self.model_names:
                        for idx, value in self.model_submission(model_name, return_dict=True, case=case, modulo=split, keep=keep, proba=proba, threshold=threshold, calibrated=calibrated).items():
                            try:
                                value = torch.from_numpy(value)
                                x, y = xy_dict[str(idx)]
                                xy_dict[str(idx)] = torch.cat([x, value], dim=0), y
                            except KeyError:
                                if self.context:
                                    img = load_image(str(idx), self.data_path+case+'/', False)
                                    img = torch.from_numpy(img).moveaxis(-1, 0)
                                    value = torch.cat([img, value])
                                xy_dict[str(idx)] = value, case + "/" + str(idx)
                        gc.collect()
                    submission_split = self.model_submission(self.meta_model_name, 
                                                             case=case,
                                                             xy_dict=xy_dict) 
                    for idx in xy_dict.keys():
                        submission.loc[int(idx), 'encoded_pixels'] = submission_split.loc[int(idx), 'encoded_pixels']
            return submission

    def plot_threshold_curve(self, predictions, ground_truths, plot=False):
        """
        Plot the threshold curve for evaluating the model.

        Parameters:
            predictions (numpy.ndarray): Model predictions.
            ground_truths (numpy.ndarray): Ground truth labels.
            metric_index (int): Index of the metric to evaluate.

        Returns:
            None
        """
        # Generate threshold values
        xs = np.arange(0.00, 1.0, 0.01)

        # Initialize variables for storing scores and best score
        scores = []
        self.best_score = 0.0

        # Iterate over each threshold value
        for x in xs:

            score = self.metric(preds=predictions, target=ground_truths.long(),threshold=x)
            score = score.cpu().numpy()

            # Update the best score and threshold if the current score is higher
            if score > self.best_score:
                self.best_score = score
                self.threshold = x
            #print(x, score)
            # Append the score to the list of scores
            scores.append(score)
        print(self.threshold, self.best_score)
        if plot:
            # Create the threshold curve plot
            plt.figure(dpi=200)
            plt.plot(xs, scores)
            plt.xlabel("Threshold value")
            plt.ylabel("Dice Coefficient")
            plt.xlim(-0.1, 1.1)
            
            # Adjust the layout and save the plot to a file
            plt.tight_layout()
            plt.show()
        return self.threshold, self.best_score
        
        
    def get_xy_dict(self, case="VALIDATION", 
                    proba=True, 
                    fold=None, 
                    calibrated=False, 
                    threshold=False, 
                    stack_4fold=False,
                    xy_dict=None,
                    video=False
                    ):
        
        submission_list = []

        gc.enable()
        for model_name in self.model_names:
            print(model_name)
            submission_list.append(self.model_submission(model_name, 
                                                         return_dict="torch", 
                                                         case=case, 
                                                         threshold=threshold, 
                                                         proba=proba, 
                                                         fold=fold, 
                                                         calibrated=calibrated,
                                                         xy_dict=xy_dict,
                                                         video=video))
            gc.collect()
            
        submission = pd.DataFrame(columns=["record_id","encoded_pixels"])
        file_list = [int(el) for el in os.listdir(self.data_path + case) if ".json" not in el]
        print(0, len(file_list))
        if fold is not None:
            file_list = [el for i, el in enumerate(file_list) if i % 10 in fold]
        print(1, len(file_list))
        submission.record_id = file_list
        submission = submission.set_index("record_id")
        

        xy_dict = {}
        
        for idx in submission.index:
            predicted_masks = []

            for i, submission_dict in enumerate(submission_list):
                if i == 0 and self.context:
                    img = load_image(str(idx), self.data_path+case+'/', False)
                    img = torch.from_numpy(img).moveaxis(-1, 0)
                    predicted_masks.append(img.cpu().to(dtype=torch.float16))
                current = submission_dict[idx].cpu().to(dtype=torch.float16)
                if stack_4fold:
                    if i % 4 == 0:
                        stack = current
                    else:
                        stack += current 
                    if i%4 == 3:
                        predicted_masks.append(stack / 4)
                    
                else:
                    predicted_masks.append(current)
                 
            x = torch.cat(predicted_masks)
            
            
            if case == "TRAIN":
                with open(os.path.join(self.data_path + case, str(idx), 'human_individual_masks.npy'), 'rb') as f:
                    y = np.load(f)
                    y = np.mean(y, axis=-1)
                    y = torch.from_numpy(y).to(dtype=torch.float16)
            else:
                with open(os.path.join(self.data_path + case, str(idx), 'human_pixel_masks.npy'), 'rb') as f:
                    y = np.load(f)
                    y = torch.from_numpy(y).to(dtype=torch.float16).moveaxis(-1, 0)
                
            xy_dict[str(idx)] = [x, y]
        return xy_dict

    
    def load_model_submission(self, model_name,
                              case='VALIDATION',
                              threshold=True, 
                              proba=True,
                              fold=None,
                              xy_dict=None,
                              calibrated=False,
                              dir_path= "/data/common/dataiku2/managed_folders/KAGGLE_V2/SoQDv7g3/"):
        print(model_name)
        dict_path = dir_path+model_name+"_dict.sav"
        try:
            model_dict = joblib.load(dict_path)
        except:
            #if True:
            model_dict = self.model_submission(model_name, return_dict="torch", case=case, 
                                                             threshold=threshold, 
                                                             proba=proba,
                                                            fold=fold,
                                                            xy_dict=xy_dict,
                                                             calibrated=calibrated
                                                            )
            joblib.dump(model_dict, dict_path)
        gc.enable()
        return model_dict
        
    def get_submission_list(self,
                            case="VALIDATION", 
                            threshold=True, 
                            proba=True,
                            fold=None,
                            xy_dict=None,
                            calibrated=False, 
                            dir_path= "/data/common/dataiku2/managed_folders/KAGGLE_V2/SoQDv7g3/"):
        
        submission_list = []
        
        for model_name in self.model_names:
            sub_submission = []
            
            if type(model_name) == list:
                for i, model_name_i in enumerate(model_name):
                    sub_submission.append(self.load_model_submission(model_name_i,
                                                                     case=case,
                                                                     proba=proba,
                                                                     fold=fold,
                                                                     xy_dict=xy_dict,
                                                                     threshold=threshold,
                                                                     calibrated=calibrated, 
                                                                     dir_path=dir_path))
                submission_list.append(sub_submission)
            else:
                submission_list.append(self.load_model_submission(model_name,
                                                                  case=case,
                                                                  proba=proba,
                                                                  fold=fold,
                                                                  xy_dict=xy_dict,
                                                                  threshold=threshold, 
                                                                  calibrated=calibrated,
                                                                  dir_path=dir_path))
        return submission_list

    def evaluate_threshold(self,
                           case="VALIDATION",
                           threshold=True,
                           proba=True,
                           calibrated=False,
                           fold=None,
                           xy_dict=None,
                           sigmoid=False,
                           dir_path= "/data/common/dataiku2/managed_folders/KAGGLE_V2/SoQDv7g3/"):

        submission_list = self.get_submission_list(
                                              case=case,
                                              fold=fold,
                                              xy_dict=xy_dict,
                                              threshold=threshold,
                                              proba=proba,
                                              calibrated=calibrated,
                                              dir_path=dir_path
        )

        submission = pd.DataFrame(columns=["record_id","encoded_pixels"])
        file_list = [int(el) for el in os.listdir(self.data_path + case) if ".json" not in el]
        if fold is not None:
            file_list = [el for i, el in enumerate(file_list) if i % 10 in fold]
        submission.record_id = file_list
        submission = submission.set_index("record_id")
        
        ground_truths = []
        predictions = []
        
        
        keys = sorted(list(submission.index))

        for idx in keys:
            predicted_mask = torch.zeros((1,256,256)).cpu()
            
            for i, submission_dict in enumerate(submission_list):
                predicted_mask += submission_dict[idx].cpu()
                 
            predicted_mask = predicted_mask / len(submission_list)
            
            if sigmoid:
                predicted_mask = torch.sigmoid(predicted_mask)
                
            predictions.append(predicted_mask)
            
            with open(os.path.join(self.data_path + case, str(idx), 'human_pixel_masks.npy'), 'rb') as f:
                y = np.load(f)
                y = torch.from_numpy(y)
            ground_truths.append(y)
        
        predictions = torch.stack(predictions)
        ground_truths = torch.stack(ground_truths)
        return self.plot_threshold_curve(predictions, ground_truths)
        

    def get_performance_combination(self, combination, submission_list, keys, 
                                    case = "VALIDATION",
                                    sigmoid=True):

        ground_truths = []
        predictions = []
        
        n = np.sum([len(model_name) if type(model_name) == list else 1 for model_name in combination])

        for idx in keys:
            predicted_mask = torch.zeros((1,256,256)).cpu()


            for model_name in combination:
                if type(model_name) == list:
                    for i, model_name_i in enumerate(model_name):
                        model_index = self.model_names.index(model_name)
                        predicted_mask += submission_list[model_index][i][idx].cpu()
                else:
                    model_index = self.model_names.index(model_name)
                    predicted_mask += submission_list[model_index][idx].cpu()

            predicted_mask = predicted_mask / n

            if sigmoid:
                predicted_mask = torch.sigmoid(predicted_mask)

            predictions.append(predicted_mask)

            with open(os.path.join(self.data_path + case, str(idx), 'human_pixel_masks.npy'), 'rb') as f:
                y = np.load(f)
                y = torch.from_numpy(y)
            ground_truths.append(y)

        predictions = torch.stack(predictions).cpu()
        ground_truths = torch.stack(ground_truths).cpu()

        xs = np.arange(0.002, 0.05, 0.002)
        scores = Parallel(n_jobs=25)(delayed(get_num_denum_img)(y_pred, y_true, xs) 
                             for y_pred, y_true in zip(predictions.numpy(), ground_truths.numpy()))
        nun_denums = np.sum(scores,axis=0)
        dices = 2*nun_denums[:,0] / nun_denums[:,1]
        #print(dices)
        max_index = np.argmax(dices)
        return dices[max_index], xs[max_index]

    def evaluate_best_combination(self,
                                  case="VALIDATION", 
                                  threshold=False, 
                                  proba=False,
                                  calibrated=True,
                                  sigmoid=True,
                                  dir_path="/home/philippe/dataiku/managed_folders/CONTRAILS/KJSQtr3w/PREDICTED/",
                                  n=4
                                 ):
        submission_list = self.get_submission_list(case=case,
                                                       threshold=threshold, 
                                                       proba=proba,
                                                       calibrated=calibrated)

    

        keys = [int(el) for el in os.listdir(self.data_path + case) if ".json" not in el]
        
        
        try:
            best_dict = joblib.load(dir_path+"best_dict.sav")
            best_score, best_threshold, best_combination = best_dict["BEST"]
        
        except FileNotFoundError:
            best_dict = {}
            best_score = 0.0
            best_threshold = None
            best_combination = None

        for n in range(n, n+1):
            
            combinations = get_combinations(self.model_names, n)



            for i, combination in enumerate(combinations):
                flatten_combination = flatten_list(combination)
                sorted_combination = sorted(flatten_combination)
                print(n, sorted_combination)
                try:
                    score, threshold = best_dict[tuple(sorted_combination)]
                    add = 'loaded'
                except:
                    score, threshold = self.get_performance_combination(combination, 
                                                                        submission_list, 
                                                                        keys, 
                                                                        case = case,
                                                                        sigmoid=sigmoid)
                    add = 'computed'
                    best_dict[tuple(sorted_combination)] = [score, threshold]
                print(score, threshold, best_score, add)
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_combination = sorted_combination
                    best_dict["BEST"] = [best_score, best_threshold, best_combination]
                joblib.dump(best_dict, dir_path+"best_dict.sav")
        return best_score, best_threshold, best_combination