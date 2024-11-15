from tensorflow.keras.models import load_model
model = load_model('/Users/jimildigaswala/Desktop/gameproject/Assets/saved_model.h5')
model.save('saved_model', save_format='tf') 