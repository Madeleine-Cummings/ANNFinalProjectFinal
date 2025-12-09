Blogpost: https://medium.com/@23dieuanhn/6e1113bbdd0e

# FIGURES
<img width="1068" height="450" alt="image" src="https://github.com/user-attachments/assets/c2581e05-b167-48e6-bb7e-017bef9c61ff" />
Shows the number of headlines per day to see if we would have a good enough range of headlines to combine for analysis

```
# LOOKING AT HEADLINES PER DAY
plt.figure(figsize=(14,6))
plt.plot(headlines_per_day["date"], headlines_per_day["num_headlines"], marker='o', linestyle='-')
plt.xlabel("Date")
plt.ylabel("Number of Headlines")
plt.title("Number of Headlines per Day")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()  # prevent label cutoff
plt.show()
```

<img width="1066" height="492" alt="image" src="https://github.com/user-attachments/assets/086a4ed3-adac-4b3e-82a8-c668a607a5c3" />

looks at how the market changes over span of 3 years; helps us identify how big of a change occurs between dates (ex: big spike around time of covid)

```
# Visualizing the market index over time

df.groupby("date")["Value"].mean().plot(figsize=(12,5))
plt.xlabel("Date")
plt.ylabel("Market Index Value")
plt.title("Market Index over Time")
plt.show()
```

<img width="2048" height="938" alt="image" src="https://github.com/user-attachments/assets/2dd00897-7b11-4c2c-8aa3-63c66fd662f8" />
Another way to visualize market index thats standardize from 0-1, making it easier to interpret the change

```
# Visualizing the market index over time

df.groupby("date")["pct_change"].mean().plot(figsize=(12,5))
plt.xlabel("Date")
plt.ylabel("Market Index Value")
plt.title("Market Index over Time")
plt.show()
```

TRAINING LOSS AND VALIDATION CURVES FOR ALL FNNS
<img width="1384" height="934" alt="image" src="https://github.com/user-attachments/assets/3d0b8a1c-df7e-429a-8489-efe179b5b75b" />
<img width="1396" height="942" alt="image" src="https://github.com/user-attachments/assets/7f67d315-0295-460b-9298-6a92dfaadb83" />

```
# Plot Training vs Validation Loss
plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('TF-IDF + FNN Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot Training vs Validation Accuracy
plt.figure(figsize=(8,5))
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('TF-IDF + FNN Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
```

<img width="688" height="470" alt="tfidffnnlosscombined" src="https://github.com/user-attachments/assets/91076143-c3c4-4666-bf49-f9461d6717a1" />
<img width="691" height="478" alt="tfidffnnacccombined" src="https://github.com/user-attachments/assets/b982529d-31ad-4e2c-9c8a-76f5d5a9a1b6" />

```
import matplotlib.pyplot as plt

# Extract metrics
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(train_loss) + 1)

# --- LOSS CURVE ---
plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title("TF-IDF + FNN Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# --- ACCURACY CURVE ---
plt.figure(figsize=(8,5))
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title("TF-IDF + FNN Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
```

<img width="700" height="475" alt="bertfnnacccombined" src="https://github.com/user-attachments/assets/fa034be0-456d-4196-a208-3f7a20b5b5d0" />
<img width="699" height="463" alt="bertfnnlosscombined" src="https://github.com/user-attachments/assets/dd3a0cef-812c-43bd-a058-d501b9b34b1e" />

```
import matplotlib.pyplot as plt

# history is returned from model_bert.fit(...)

# Extract accuracy and loss curves
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(train_loss) + 1)

# LOSS CURVE
plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('BERT + FNN Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# ACCURACY CURVE
plt.figure(figsize=(8,5))
plt.plot(epochs, train_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title('BERT + FNN Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
```
<img width="686" height="423" alt="avg%change" src="https://github.com/user-attachments/assets/55777863-20b2-48af-86d4-da8c0784ac2e" />
Visualization of the confusion matrix - looking at true positives % change and false negatives % change to see if there's a difference between truly identified vs falsely identified headlines

```
import matplotlib.pyplot as plt

# Compute averages from the actual groups
avg_changes = {
    "TP": df_tp['pct_change'].mean(),
    "TN": df_tn['pct_change'].mean(),
    "FP": df_fp['pct_change'].mean(),
    "FN": df_fn['pct_change'].mean()
}

# Plot
plt.figure(figsize=(8,5))
plt.bar(avg_changes.keys(), avg_changes.values(), color=["green","blue","orange","red"])
plt.ylabel("Average pct_change")
plt.title("Average Percent Change by Confusion Matrix Group")
plt.show()
```
