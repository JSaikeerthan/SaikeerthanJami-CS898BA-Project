"""
Main entry point for the project.
"""

from src.utils.dataset import load_dataset


def main():

    images, labels, class_names = load_dataset()

    print("\nFirst five classes:")

    for name in class_names[:5]:
        print(name)


if __name__ == "__main__":
    main()