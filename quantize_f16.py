
import util.model_util as model_util
import util.dataset_util as dataset_util



import torch


from hyperparameters import DEVICE

model_path = r'models\resnet_model_trained.pt'
model_path_half = r'models\resnet_model_trained_fp16.pt'




try:
    model = model_util.load_pytorch_script_model(model_path)

    # You can now use the model for inference or further processing
    print("Model is ready for use.")
except:
    print("Model could not be loaded.")




train_dataset, test_dataset, train_loader, test_loader =dataset_util.dataset_load()



with torch.no_grad(): # save memory during inference
    print('Test accuracy: %.2f%%' % (model_util.compute_accuracy_half(model, test_loader, device=DEVICE)))


try:

    model_util.save_model_script(model.half(), model_path_half)
    print('model 16bit saved')

except:
    print('ERROR')
