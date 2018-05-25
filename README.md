# ResNet50train
# ResNet50train201805250900
The model is trained with the initial weights loaded from the model trained in ImageNet. The final dense layer is removed and replaced with dense layer of size 10 which is the total number of the output classes of camera models in our dataset. The last 14 layers were trained and other layers were kept frozen for the training. The batch size was fixed to 30 and trained for 100 epochs.
