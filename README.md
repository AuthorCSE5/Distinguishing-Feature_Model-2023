For Blade-Chest inner model find the code at https://github.com/csinpi/blade_chest

For GNNRank algorithm find the codes at https://github.com/SherylHYX/GNNRank/tree/14c5b3e22ed4dc76639013580359809885b0d0fa

## Synthetic Data 

1) Run RC_SF_DFlearn_DFdata.py/RC_SF_DFlearn_BTLdata.py/RC_SF_DFlearn_SFdata.py for DF model data/ BTL model data/ SF model data respectively.

2) Run Dota_RC_SF_DF.py for all the real datasets.

3) For Blade-Chest inner model, the following lines have been added to the original code in the Testing part of the Final Test section to extract the data ids used for testing. These ids will be used in BC_Accuracy.py to evaluate the prediction accuracy and rmse values for the real datsets. 

      FILE *fp;
      
      fp = fopen("test_indices_BC.txt", "a");

			fprintf(fp, "%d \n", triple_id);

			fclose(fp);
      
4) 

