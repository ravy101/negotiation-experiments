import translation_model
import sys


def main() -> int:
    """Echo the input arguments to standard output"""
    translator = translation_model.TranslationModel()
    while True:
        print("Enter text:")
        text = input()
        if len(text) >= 1:
            tranlation = translator.translate(text)
            print("Translation:")
            print(tranlation)
        else:
            break
    return 0

if __name__ == '__main__':
    sys.exit(main())