For Blade-Chest inner model find the code at https://github.com/csinpi/blade_chest

For GNNRank algorithm find the codes at https://github.com/SherylHYX/GNNRank/tree/14c5b3e22ed4dc76639013580359809885b0d0fa

1) Run RC_SF_DFlearn_DFdata.py/RC_SF_DFlearn_BTLdata.py/RC_SF_DFlearn_SFdata.py for DF model data/ BTL model data/ SF model data respectively.

2) Run Dota_RC_SF_DF.py for all the real datasets.

3) For Blade-Chest inner model, the following lines have been added to the original code in the Testing part of the Final Test section to extract the data ids used for testing. These ids will be used in BC_Accuracy.py to evaluate the prediction accuracy and rmse values for the real datsets. Find the updated BC.c file in Blade_Chest/ folder here.

      FILE *fp;
      
      fp = fopen("test_indices_BC_WoL.txt", "a");

      fprintf(fp, "%d \n", triple_id);

      fclose(fp);
      
4) ./BC -d 50 -l 0.001 -M 2 -S 5 ../datasets/starcraft/WoL.txt BC_WoL_5.txt ---- execute this command for seed value 5 and save the model parameters in a file BC_WoL_5(seed value).txt for WoL dataset.

5) ./BC -d 50 -l 0.001 -M 2 ../datasets/BC_DFdata_train664_0.txt 664_0.txt ----- execute this command for the synthetic dataset BC_DFdata_train664_0.txt for seed value 0 and save the model parameters in a file 664_0(seed value).txt. Save the synthetic datasets in the blade_chest/datasets/ folder as BC_DFdata_train664_0.txt for dataset generated from the Distinguishing Features (DF) model for number of training pairs = 664(c = 1) and for seed value 0.

6) For GNNRank, we need to extract the scores of items from the algorithm that are used for further ranking purpose. It is because, in our code, we need the scores explicitly to evaluate the prediction accuracy as well as the rmse values. Hence, there are slight changes in the original GNNRank/src/train.py file (https://github.com/SherylHYX/GNNRank/tree/14c5b3e22ed4dc76639013580359809885b0d0fa) . Find the updated file in the GNNRank/ folder here.

7) Create the finer versions of the real/synthetic datasets in the original GNNRank/data/ folder. Create a data preprocessing file named DF_preprocess.py in the same folder to find the adjacency matrix of the datasets. The finer versions of the datasets are created using the GNNRank_Data.py file.

8) The scores will be saved as Score_Dist.txt, Score_innerproduct.txt, Score_proximalbaseline.txt, Score_proximaldist.txt, Score_proximalinnerproduct.txt in the original GNNRank/src/ folder(https://github.com/SherylHYX/GNNRank/tree/14c5b3e22ed4dc76639013580359809885b0d0fa). The score text files for synthetic dataset generated from the DF model for number of pairs = 664 and seed value 0,  need to be placed in the GNNRank_Datasets_finer/Synthetic/DF/664/0 folder. The score text files for real dataset WoL and for seed value 0, need to be saved in GNNRank_Datasets_finer/WoL/0 folder.

9) The data for Distinguishing Feature(DF) model is generated using pairwise_comparisonsC.py, data for BTL model is generated using Data_BTL_model.py, data for Salient Features(SF) model is generated using SFmodel_Data.py.
 - The embeddings for the items are generated using gen_embedding.py in both DF and SF model.
 - The weights in SF models are generated using SF_model_weights.py.
 - Scores for BTL model are generated using Generate_Score_BTL.py.

