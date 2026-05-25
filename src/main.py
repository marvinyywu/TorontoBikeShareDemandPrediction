import preprocess
import train

if __name__ == "__main__":
    preprocessed_df = preprocess.preprocess_data()
    model = train.train_model(preprocessed_df)    