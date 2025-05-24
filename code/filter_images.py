import os
import shutil
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path  # Using pathlib for easier path manipulation


class PoseDatasetProcessor:
    CLASS_MAPPING = {
        'sitting': [
            'sitting quietly',
            'sitting, talking in person, on the phone, computer, or text messaging, light effort'
            # Add more MPII activities corresponding to 'sitting'
        ],
        'standing': [
            'paddle boarding, standing',
            'standing, doing work'
            # Add more MPII activities corresponding to 'standing'
        ],
        'walking': [
            'walking, for exercise, with ski poles',  # Corrected 'exercise'
            'skating, ice dancing'
            # Add more MPII activities corresponding to 'walking' or movement
        ]
    }

    # SPLIT_RATIOS are implicit in the train_test_split logic now
    CLASS_DIRS = ['standing', 'sitting', 'walking']  # Target classes

    def __init__(self, base_path='../data'):
        self.base_path = Path(base_path)  # Use pathlib.Path
        self.raw_images_path = self.base_path / 'images'
        self.annotations_path = self.base_path / 'mpii_human_pose_v1_u12_1.mat'
        self.processed_path = self.base_path / 'processed_mpii_poses'  # More descriptive name
        self.plots_path = self.processed_path / 'plots'  # Directory for saving plots
        self.df = None

        self._validate_paths()
        self.plots_path.mkdir(parents=True, exist_ok=True)  # Create plots directory

    def _validate_paths(self):
        """Validate existence of essential files and directories"""
        if not self.annotations_path.exists():
            raise FileNotFoundError(
                f"MAT file not found at: {self.annotations_path}\n"
                "Please ensure:\n"
                f"1. The '{self.base_path.name}' directory exists relative to your script or as an absolute path.\n"
                f"2. The '{self.annotations_path.name}' file is in the '{self.base_path.name}' directory."
            )

        if not self.raw_images_path.exists() or not self.raw_images_path.is_dir():
            raise NotADirectoryError(
                f"Images directory not found or is not a directory: {self.raw_images_path}\n"
                f"Please create '{self.raw_images_path}' and place the MPII images there."
            )

    def load_and_filter_data(self):
        """Load and filter data from MPII dataset"""
        print("Loading annotations...")
        mat_data = scipy.io.loadmat(self.annotations_path)
        release = mat_data['RELEASE'][0, 0]
        annolist = release['annolist'][0]

        data = []
        print("Processing annotations and filtering images by activity...")
        for idx, entry in enumerate(annolist):
            try:
                img_name = entry['image'][0, 0]['name'][0]
                activity_struct = release['act'][idx, 0]
                activity = self._extract_activity_name(activity_struct)
            except (IndexError, KeyError, TypeError):  # Added TypeError for safety
                # print(f"Skipping entry {idx} due to missing data.")
                continue  # Skip if essential data is missing

            # Normalize activity name for robust matching
            normalized_activity = activity.strip().lower()

            # Check if this activity maps to one of our target classes
            mapped_class = self._map_to_class(normalized_activity)
            if mapped_class:  # Only add if it maps to a target class
                data.append({
                    'image_name': img_name,
                    'activity': normalized_activity,  # Store normalized activity
                    'class': mapped_class
                })

        if not data:
            print("No images found matching the activities in CLASS_MAPPING. Please check your mapping.")
            self.df = pd.DataFrame()  # Ensure df is an empty DataFrame
            return

        self.df = pd.DataFrame(data)
        # Drop duplicates based on image_name, keeping the first class assignment for an image
        # This means if an image has activities mapping to 'sitting' and 'standing',
        # it will be assigned to the class of the first encountered activity.
        # If you want to assign to all relevant classes, this logic would need to change (e.g., explode rows).
        self.df = self.df.drop_duplicates(subset=['image_name'], keep='first')
        print(f"Loaded and filtered {len(self.df)} unique images for target classes.")

    def _extract_activity_name(self, activity_struct):
        """Extract activity name from MAT structure"""
        if 'act_name' in activity_struct.dtype.names:
            act_name_field = activity_struct['act_name']
            # act_name_field is often an array containing a single string, e.g., array(['activity name'], dtype='<U...')
            if isinstance(act_name_field, np.ndarray) and act_name_field.size > 0:
                # Access the first element, which should be the string
                name_str = act_name_field[0]
                if isinstance(name_str, str):
                    return name_str
        return ''  # Return empty string if not found or not in expected format

    def _map_to_class(self, activity):
        """Map activities to simplified classes"""
        for cls, activities_list in self.CLASS_MAPPING.items():
            if activity in activities_list:
                return cls
        return None  # Return None if no mapping found

    def create_directory_structure(self):
        """Create processed directory structure for train/val/test splits"""
        if self.df is None or self.df.empty:
            print("DataFrame is empty. Skipping directory structure creation.")
            return

        print(f"Creating directory structure under: {self.processed_path}")
        for class_dir in self.CLASS_DIRS:  # Use self.CLASS_DIRS which contains the target classes
            for split in ['train', 'val', 'test']:  # Define splits directly
                path = self.processed_path / class_dir / split
                path.mkdir(parents=True, exist_ok=True)
        print("Directory structure created.")

    def split_and_save_dataset(self):
        """Split and save dataset into directories"""
        if self.df is None or self.df.empty or 'class' not in self.df.columns:
            print("DataFrame is not properly loaded or 'class' column is missing. Cannot split.")
            return

        # Ensure there are samples to stratify on, and more than 1 class for stratification
        if self.df['class'].nunique() < 2 and len(self.df) > 1:
            print("Warning: Stratification requires at least 2 classes. Performing non-stratified split.")
            stratify_col = None
        elif self.df['class'].nunique() == 0 and len(self.df) > 0:
            print("Warning: No classes assigned, cannot stratify.")  # should not happen if load_and_filter is ok
            stratify_col = None
        elif self.df.empty:
            print("DataFrame is empty. Cannot split.")
            return
        else:
            stratify_col = self.df['class']

        # Define split ratios directly
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1  # test_ratio is 1 - train_ratio - val_ratio

        # Split into train and temp (val + test)
        # Ensure that test_size for the first split is val_ratio + test_ratio
        try:
            train_df, temp_df = train_test_split(
                self.df,
                test_size=(val_ratio + test_ratio),
                stratify=stratify_col,
                random_state=42
            )
            # Split temp into val and test
            # test_size for the second split is test_ratio / (val_ratio + test_ratio)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(test_ratio / (val_ratio + test_ratio)),
                stratify=temp_df['class'] if stratify_col is not None else None,
                # Stratify temp_df if original was stratified
                random_state=42
            )
        except ValueError as e:
            print(f"Error during train_test_split (likely too few samples for stratification): {e}")
            print("Attempting non-stratified split...")
            train_df, temp_df = train_test_split(self.df, test_size=(val_ratio + test_ratio), random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=(test_ratio / (val_ratio + test_ratio)),
                                               random_state=42)

        print(f"\n🏁 Starting dataset processing and image copying...")
        print(f"🔧 Root directory for processed data: {self.processed_path}")

        # Save images with progress reporting
        self._save_split_images(train_df, 'train')
        self._save_split_images(val_df, 'val')
        self._save_split_images(test_df, 'test')

        # Final summary
        print("\n🎉 Dataset organization completed!")
        print(f"📂 Total unique images processed for defined classes: {len(self.df)}")
        print(f"📂 Final directory structure for splits:")
        print(f"  - Training:   {self.processed_path / '[class]' / 'train'}")
        print(f"  - Validation: {self.processed_path / '[class]' / 'val'}")
        print(f"  - Testing:    {self.processed_path / '[class]' / 'test'}")
        print(f"\nReplace '[class]' with actual class names: {', '.join(self.CLASS_DIRS)}")

    def _save_split_images(self, df_split, split_name):
        """Save images to corresponding split directory"""
        if df_split.empty:
            print(f"\nNo images to save for {split_name} split.")
            return

        saved_count = 0
        class_counts = {cls: 0 for cls in self.CLASS_DIRS}

        print(f"\nCopying images for {split_name} split...")
        for _, row in df_split.iterrows():
            src = self.raw_images_path / row['image_name']
            # Ensure 'class' column exists and has a valid value before creating destination path
            if 'class' not in row or pd.isna(row['class']):
                print(f"  Warning: Skipping image {row['image_name']} due to missing or invalid class.")
                continue

            dst_dir = self.processed_path / row['class'] / split_name
            # dst_dir.mkdir(parents=True, exist_ok=True) # Already created in create_directory_structure
            dst = dst_dir / row['image_name']

            if src.exists():
                try:
                    shutil.copy2(src, dst)  # Use copy2 to preserve metadata
                    saved_count += 1
                    if row['class'] in class_counts:  # Check if class is expected
                        class_counts[row['class']] += 1
                    else:
                        print(
                            f"  Warning: Image {row['image_name']} has unexpected class '{row['class']}' during saving.")
                except Exception as e:
                    print(f"  Error copying {src} to {dst}: {e}")
            else:
                print(f"  Warning: Source image not found, cannot copy: {src}")

        if saved_count > 0:
            print(f"✅ Successfully saved {saved_count} images to {split_name} split.")
            print(f"📁 Location: {self.processed_path / '[class]' / split_name}".replace('[class]', '{class}'))
            print("📊 Class distribution for this split:")
            for cls, count in class_counts.items():
                if count > 0:  # Only print classes that actually have images in this split
                    print(f"  - {cls.capitalize()}: {count} images")
        else:
            print(
                f"No images were saved for the {split_name} split (source files might be missing or all rows skipped).")

    def plot_class_distribution(self):
        """Visualize class distribution and save the plot"""
        if self.df is None or self.df.empty or 'class' not in self.df.columns:
            print("DataFrame is not loaded or 'class' column is missing. Cannot plot distribution.")
            return

        counts = self.df['class'].value_counts()
        if counts.empty:
            print("No data to plot for class distribution.")
            return

        plt.figure(figsize=(8, 6))  # Create a new figure
        counts.plot(kind='bar', color=['#4C72B0', '#55A868', '#C44E52'])  # Example colors
        plt.title('Class Distribution of Filtered MPII Poses')
        plt.xlabel('Pose Class')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=0)
        plt.tight_layout()  # Adjust layout to prevent labels overlapping

        plot_filename = "class_distribution.png"
        save_path = self.plots_path / plot_filename
        try:
            plt.savefig(save_path)
            print(f"\n📊 Class distribution plot saved to: {save_path}")
            plt.close()  # Close the plot to free memory and prevent display
        except Exception as e:
            print(f"Error saving class distribution plot: {e}")
            plt.show()  # Fallback to showing if saving fails

    def plot_class_examples(self, num_examples=1):
        """Display example images for each class and save the plot"""
        if self.df is None or self.df.empty or 'class' not in self.df.columns:
            print("DataFrame is not loaded or 'class' column is missing. Cannot plot examples.")
            return

        unique_classes = sorted(self.df['class'].dropna().unique())  # Get sorted unique classes
        if not unique_classes:
            print("No classes with data found to plot examples.")
            return

        num_classes = len(unique_classes)
        fig, axs = plt.subplots(num_examples, num_classes, figsize=(5 * num_classes, 5 * num_examples), squeeze=False)

        print("\n🖼️ Generating class example plot...")
        for i, cls in enumerate(unique_classes):
            class_samples_df = self.df[self.df['class'] == cls]
            if class_samples_df.empty:
                for k_ex in range(num_examples):
                    axs[k_ex, i].set_title(f"{cls.capitalize()}\n(No Samples)")
                    axs[k_ex, i].axis('off')
                continue

            # Take up to num_examples samples, or fewer if not enough available
            samples_to_plot = class_samples_df.sample(min(num_examples, len(class_samples_df)), random_state=42)

            for k_ex, (_, sample_row) in enumerate(samples_to_plot.iterrows()):
                img_path = self.raw_images_path / sample_row['image_name']
                ax_current = axs[k_ex, i]
                if img_path.exists():
                    try:
                        img = plt.imread(img_path)
                        ax_current.imshow(img)
                        ax_current.set_title(f"{cls.capitalize()} (Example {k_ex + 1})")
                    except Exception as e_img:
                        ax_current.set_title(f"{cls.capitalize()}\nError loading image:\n{img_path.name}", fontsize=8)
                        print(f"  Error loading image {img_path}: {e_img}")
                else:
                    ax_current.set_title(f"{cls.capitalize()}\nImage not found:\n{img_path.name}", fontsize=8)
                    print(f"  Warning: Example image not found: {img_path}")
                ax_current.axis('off')

        plt.tight_layout()
        plot_filename = "class_examples.png"
        save_path = self.plots_path / plot_filename
        try:
            plt.savefig(save_path)
            print(f"🖼️ Class examples plot saved to: {save_path}")
            plt.close()  # Close the plot
        except Exception as e:
            print(f"Error saving class examples plot: {e}")
            plt.show()  # Fallback


if __name__ == "__main__":
    try:
        print("Initializing Pose Dataset Processor...")
        processor = PoseDatasetProcessor(base_path='../data')  # Ensure this path is correct

        # Data processing pipeline
        processor.load_and_filter_data()

        if processor.df is not None and not processor.df.empty:
            processor.create_directory_structure()
            processor.split_and_save_dataset()

            # Data analysis and visualization (saving plots instead of showing)
            processor.plot_class_distribution()
            processor.plot_class_examples(num_examples=1)  # Plot 1 example per class
        else:
            print("No data loaded or filtered. Exiting.")

    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR - FILE NOT FOUND: {e}")
        print("Please ensure the .mat file and images directory are correctly placed.")
    except NotADirectoryError as e:
        print(f"\nCRITICAL ERROR - DIRECTORY NOT FOUND: {e}")
        print("Please ensure the images directory exists.")
    except Exception as e:
        import traceback

        print(f"\nAN UNEXPECTED ERROR OCCURRED: {str(e)}")
        print("Traceback:")
        traceback.print_exc()