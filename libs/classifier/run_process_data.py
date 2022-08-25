from classifier.utils.processing import DataProcessor

if __name__ == "__main__":
    dp = DataProcessor(save_folder="/home/anhtt163/PycharmProjects/outsource/dataset/phone/classifier")
    dp.process_folder(folder_path="/home/anhtt163/PycharmProjects/outsource/dataset/phone/1class/train")
