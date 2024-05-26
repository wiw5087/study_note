## Takeaways from dog_cat.ipynb
In this exercise, a few things were tried out:

1. How to process pictures to be ready for image processing:
Read picture files -> Decode the JPEG content to RGB grids of pixels -> Convert into floating point tensors -> Rescale the pixel values (0 to 255) to [0,1] interval (normalization)

2. How to use data augmentation to address overfitting problems

3. Dropout layer is only applied for fully connected layers

4. Three different ways to use a pretrained model:
- All methods need to remove the fully connected layers
- Use pretrained model to predict features on the current dataset. Then add a fully connected layers to train based on these predicted features. Notice that data augmentation cannot be used in this process because the pretrained model is only make a prediction on the actual images.
- Extend the base model and run end to end on the inputs. Freeze the base model, not allow it to be retrained, only train the fully connected layers. Data augemntation can be used in this method because when retraining, each input image is read multiple times through epochs. 
- Fine tuning the base model. 
Fine-tuning consists of unfreezing a few of the top layers of a frozen model base used for feature extraction, and jointly training both the newly added part of the model (fully connected classifier) and these top layers. 

It's only possible to fine tune the top layers of hte conv base once the classifier on top has already been trained. If the classifier isn't already trained, then the error signal propagating through the network during training will be too large, and the representations previously learned by the layers being fine-tuned will be destroyed. 

Fine-Tuning flow: 
1. Add fully connected layers on top of a pretrained base network 
2. Freeze the base network 
3. Train the part you added 
4. Unfreeze some layers in the base network 
5. Jointly train both layers and the part you added

Why only train top layers:
1. Earlier layers in conv base encode more generic, reusable features.
2. The more parameters you are training, the more you are at risk of overfitting. 