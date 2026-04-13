import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

print("=" * 50)
print("GESTURE MODEL TRAINER")
print("=" * 50)

# Load dataset
print("\n✓ Loading dataset...")
try:
    with open('gestos_dataset.pkl', 'rb') as f:
        gestos = pickle.load(f)
except FileNotFoundError:
    print("ERROR: gestos_dataset.pkl not found!")
    print("Run grabar_gestos.py first to record gestures")
    exit()

# Prepare data
print("✓ Preparing data...")
X = []
y = []

for gesto_nombre, ejemplos in gestos.items():
    if len(ejemplos) > 0:
        print(f"  {gesto_nombre}: {len(ejemplos)} examples")
        for ejemplo in ejemplos:
            X.append(np.array(ejemplo).flatten())
            y.append(gesto_nombre)
    else:
        print(f"  {gesto_nombre}: 0 examples (skipped)")

X = np.array(X)
y = np.array(y)

print(f"\n✓ Total examples: {len(X)}")
print(f"✓ Number of gestures: {len(np.unique(y))}")

if len(X) == 0:
    print("\nERROR: No examples to train!")
    exit()

# Normalize data
print("\n✓ Normalizing data...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model
print("✓ Training RandomForest model...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo.fit(X, y)

# Calculate accuracy
accuracy = modelo.score(X, y)
print(f"✓ Training accuracy: {accuracy:.2%}")

# Save model
print("\n✓ Saving model...")
with open('gesto_modelo.pkl', 'wb') as f:
    pickle.dump((modelo, scaler), f)

print("\n" + "=" * 50)
print("✓ Model saved successfully!")
print("=" * 50)
print(f"\nModel info:")
print(f"  Gestures: {list(np.unique(y))}")
print(f"  Accuracy: {accuracy:.2%}")
print(f"  File: gesto_modelo.pkl")
print("=" * 50)
print("\nNext step: Run main.py for real-time prediction")
print("=" * 50)