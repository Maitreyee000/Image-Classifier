# Streamlit-Image-Classifier in google colab

For this demo, I will be using the VGG19 image classification model. The image classifier has been built to classify images of Sea, Glaciers, Mountain, Forest, Street, and Buildings

![](https://github.com/poojatambe/Streamlit-Image-Classifier/blob/main/streamlit.gif)

# Steps:

1. Install streamlit.
```
pip install -U streamlit
```

2. Install localtunnel.
```
npm install localtunnel
```

3. Run python file on streamlit.
```
streamlit run streamlit_vgg.py &>/dev/null&
```

4. Specify PORT and click on the link generated to run streamlit.
```
npx localtunnel --port 8501
```

# Reference:
https://discuss.streamlit.io/t/free-streamlit-dev-environment-through-colaboratory/2778/11
