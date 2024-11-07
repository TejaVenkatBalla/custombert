import logging
import sys
from classifier import NeuraGuard

def main():
    """Main function to run the Cybercrime Classifier."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        classifier = NeuraGuard.load_model("multioutput_bert_model.joblib")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    while True:
        print("\nEnter a cybercrime description (or 'quit' to exit):")
        user_input = input().strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            break

        if not user_input:
            print("Please enter a valid description.")
            continue

        try:
            prediction = classifier.predict(user_input)

            print("\n--- Prediction Results ---")
            print(f"Category: {prediction['category']} (Confidence: {prediction['category_confidence']:.2%})")
            print(f"Subcategory: {prediction['subcategory']} (Confidence: {prediction['subcategory_confidence']:.2%})")

        except Exception as e:
            print(f"Prediction error: {e}")

    print("\nThank you for using the NeuraGuard!")

if __name__ == "__main__":
    main()
