with open('src/multical/preprocessing/pipeline.py') as f:
    text = f.read()

text = text.replace("print(f\"Applying {method}\")", "print(f\"Applying {method}, shape before={absor.shape}\")")
text = text.replace("step_plot = step[-1]", "print(f\"shape after={absor.shape}\"); step_plot = step[-1]")

with open('src/multical/preprocessing/pipeline.py', 'w') as f:
    f.write(text)
