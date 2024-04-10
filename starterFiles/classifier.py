from taipy.gui import Gui
from tensorflow.keras import models
from PIL import Image
import numpy as np


class_names = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}


model= models.load_model("keshav_baseline.keras")

def predict_image(model,path_to_img):
    img =Image.open(path_to_img)
    img = img.convert("RGB")
    img = img.resize((32,32))
    data=np.asarray(img)
    data/255
    print("after",data[0][0])
    probs=model.predict(np.array([data])[:1])
    top_prob=probs.max()
    top_pred=class_names[np.argmax(probs)]

    return top_prob,top_pred


content=""
img_path="placeholder_image.png"
prob=0
pred=""
index="""
<|text-center|
<|{"logo.png"}|image|width=25vw|>

<|{content}|file_selector|extensions=.png|>
select an image from your file system


<|{pred}|>


<|{img_path}|image|width=25vw|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>

>
"""



def on_change(state,var_name,var_value):
    if var_name == "content":
        top_prob, top_pred = predict_image(model,var_value)
        state.prob = round(top_prob * 100)
        state.pred = "this is a " + top_pred
        state.img_path = var_value
    print(var_name,var_value)

app=Gui(page=index)

if __name__=="__main__":
    app.run(use_reloader=True)