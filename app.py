import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pandas as pd
import altair as alt



@st.cache(allow_output_mutation=True)
def load_mnist_model():
    try:
        model = load_model('final_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
# Load the trained model
model = load_model('final_model.h5')

# Streamlit app configuration
st.title("MNIST Digit Classifier")
st.markdown("""
Draw a digit on the canvas below and see the model predict the digit!
""")

st.sidebar.header("Configuration")

# Specify brush parameters and drawing mode
b_color = st.sidebar.color_picker("Brush color", "#000000")
bg_color = st.sidebar.color_picker("Background color", "#FFFFFF")
drawing_mode = st.sidebar.checkbox("Drawing mode?", True)

# Create a canvas component
canvas_result = st_canvas(
    stroke_width=20,
    stroke_color=b_color,
    background_color=bg_color,
    height=280,
    width=280,
    drawing_mode='freedraw' if drawing_mode else 'transform',
    key="canvas"
)

def preprocess_image(image_data):
    # Convert to PIL image
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA').convert('L')
    # Resize to 28x28 pixels
    img = img.resize((28, 28))
    # Invert colors (white background to black background)
    img = Image.eval(img, lambda x: 255 - x)
    # Normalize to range 0-1
    img = np.array(img).astype('float32') / 255.0
    # Reshape to match model input
    img = img.reshape(1, 28, 28, 1)
    return img

if canvas_result.image_data is not None:
    # Preprocess the canvas image data
    img = preprocess_image(canvas_result.image_data)
    
    # Make prediction
    prediction = model.predict(img)
    pred_digit = np.argmax(prediction)
    probabilities = prediction[0]
    
    st.write(f"Predicted digit: {pred_digit}")
    
    # # Display the probabilities for each class
    # for i, prob in enumerate(probabilities):
    #     st.write(f"Probability of {i}: {prob:.4f}")

        # Create a dataframe for probabilities
    prob_df = pd.DataFrame(probabilities, index=range(10), columns=["Probability"])
    
    # Create a horizontal bar chart
    st.bar_chart(prob_df)
    # st.dataframe(prob_df.style.set_properties(**{'background-color': 'white', 'color': 'black'}))

else:
    st.write("Please draw a digit on the canvas.")


