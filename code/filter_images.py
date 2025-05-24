import os
import shutil
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class PoseDatasetProcessor:
    CLASS_MAPPING = {
        'sitting': [
            'sitting quietly',
            'sitting, talking in person, on the phone, computer, or text messaging, light effort'
        ],
        'standing': [
            'paddle boarding, standing',
            'standing, doing work'
        ],
        'walking': [
            'walking, for exercise, with ski poles',
            'skating, ice dancing'
        ]
    }
    
    SPLIT_RATIOS = {'train': 0.7, 'val': 0.2, 'test': 0.1}
    CLASS_DIRS = ['standing', 'sitting', 'walking']

    def __init__(self, base_path='../data'):
        self.base_path = base_path
        self.raw_images_path = os.path.join(base_path, 'images')
        self.annotations_path = os.path.join(base_path, 'mpii_human_pose_v1_u12_1.mat')
        self.processed_path = os.path.join(base_path, 'processed')
        self.df = None

        # Validate critical paths
        self._validate_paths()

    def _validate_paths(self):
        """Validate existence of essential files and directories"""
        if not os.path.exists(self.annotations_path):
            raise FileNotFoundError(
                f"MAT file not found at: {self.annotations_path}\n"
                "Please ensure:\n"
                "1. The 'data' directory exists in the script's parent directory\n"
                "2. The 'mpii_human_pose_v1_u12_1.mat' file is in the 'data' directory"
            )

        if not os.path.exists(self.raw_images_path):
            raise NotADirectoryError(
                f"Images directory not found: {self.raw_images_path}\n"
                "Please create 'data/images' and place the images there"
            )

    def load_and_filter_data(self):
        """Load and filter data from MPII dataset"""
        mat_data = scipy.io.loadmat(self.annotations_path)
        release = mat_data['RELEASE'][0,0]
        annolist = release['annolist'][0]
        
        data = []
        for idx, entry in enumerate(annolist):
            try:
                img_name = entry['image'][0,0]['name'][0]
                activity = self._extract_activity_name(release['act'][idx,0])
            except (IndexError, KeyError):
                continue

            if activity in sum(self.CLASS_MAPPING.values(), []):
                data.append({
                    'image_name': img_name,
                    'activity': activity,
                    'class': self._map_to_class(activity)
                })

        self.df = pd.DataFrame(data).drop_duplicates('image_name')

    def _extract_activity_name(self, activity_struct):
        """Extract activity name from MAT structure"""
        if 'act_name' in activity_struct.dtype.names:
            act_name = activity_struct['act_name']
            if isinstance(act_name, np.ndarray) and act_name.size > 0:
                return act_name[0]
        return ''

    def _map_to_class(self, activity):
        """Map activities to simplified classes"""
        for cls, activities in self.CLASS_MAPPING.items():
            if activity in activities:
                return cls
        return ''

    def create_directory_structure(self):
        """Create processed directory structure"""
        for class_dir in self.CLASS_DIRS:
            for split in self.SPLIT_RATIOS:
                path = os.path.join(self.processed_path, class_dir, split)
                os.makedirs(path, exist_ok=True)

    def split_and_save_dataset(self):
        """Split and save dataset into directories"""
        # Stratified split
        train_df, temp_df = train_test_split(
            self.df, test_size=0.3, stratify=self.df['class'], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.33, stratify=temp_df['class'], random_state=42
        )

        print(f"\n🏁 Starting dataset processing")
        print(f"🔧 Root directory: {self.processed_path}")

        # Save images with progress reporting
        self._save_split_images(train_df, 'train')
        self._save_split_images(val_df, 'val')
        self._save_split_images(test_df, 'test')

        # Final summary
        print("\n🎉 Dataset organization completed!")
        print(f"📂 Total processed images: {len(self.df)}")
        print(f"📂 Final directory structure:")
        print(f"  - Training:   {os.path.join(self.processed_path, '[class]', 'train')}")
        print(f"  - Validation: {os.path.join(self.processed_path, '[class]', 'val')}")
        print(f"  - Testing:    {os.path.join(self.processed_path, '[class]', 'test')}")
        print(f"\nReplace '[class]' with actual class names: {', '.join(self.CLASS_DIRS)}")

    def _save_split_images(self, df, split_name):
        """Save images to corresponding split directory"""
        saved_count = 0
        class_counts = {cls: 0 for cls in self.CLASS_DIRS}

        for _, row in df.iterrows():
            src = os.path.join(self.raw_images_path, row['image_name'])
            dst = os.path.join(
                self.processed_path,
                row['class'],
                split_name,
                row['image_name']
            )

            if os.path.exists(src):
                shutil.copy(src, dst)
                saved_count += 1
                class_counts[row['class']] += 1

        # Print summary for this split
        print(f"\n✅ Successfully saved {saved_count} images to {split_name} split")
        print(f"📁 Location: {os.path.join(self.processed_path, '[class]', split_name).replace('[class]', '{class}')}")
        print("📊 Class distribution:")
        for cls, count in class_counts.items():
            if count > 0:
                print(f"  - {cls.capitalize()}: {count} images")

    def plot_class_distribution(self):
        """Visualize class distribution"""
        counts = self.df['class'].value_counts()
        counts.plot(kind='bar', color=['#4C72B0', '#55A868', '#C44E52'])
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.show()

    def plot_class_examples(self):
        """Display example images for each class"""
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for i, cls in enumerate(self.CLASS_MAPPING.keys()):
            sample = self.df[self.df['class'] == cls].sample(1)
            img_path = os.path.join(self.raw_images_path, sample['image_name'].iloc[0])
            img = plt.imread(img_path)
            axs[i].imshow(img)
            axs[i].set_title(cls.capitalize())
            axs[i].axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    try:
        processor = PoseDatasetProcessor()
        
        # Data processing pipeline
        processor.load_and_filter_data()
        processor.create_directory_structure()
        processor.split_and_save_dataset()
        
        # Data analysis and visualization
        processor.plot_class_distribution()
        processor.plot_class_examples()

    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Please check directory structure!")
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {str(e)}")