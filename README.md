# Imperceptible-Adversarial-Sample-Generation
The code for the upsampling and information embedding image processing steps is based on Hiding Multiple Images into a Single Image Using Up-Sampling study was improved, but the author's code is not open source and cannot be shared here, but the processed CUTE80 dataset is given
first step 
pip install -r requirements.txt
then using
 python main.py --image_folder ./data/cropped-CUTE80 --perturb_folder ./data/CUTE80-upsample --saved_model STR_modules/downloads_models/STARNet-TPS-ResNet-BiLSTM-CTC-sensitive.pth --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC
You can change the model by changing the corresponding name on the command line: --Transformation,--FeatureExtraction,--SequenceModeling,--Prediction
