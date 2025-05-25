import os
import shutil
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path  # Using pathlib for easier path manipulation


class PoseDatasetProcessor:
    # CLASS_MAPPING will remain, but load_and_filter_data will use a specific map
    # based on the provided script.
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
    CLASS_DIRS = ['standing', 'sitting', 'walking']  # Target classes - these match the new mapping's output

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
        """Load and filter data from MPII dataset based on a specific activity list and mapping."""
        print("Loading annotations...")
        mat_data = scipy.io.loadmat(self.annotations_path)
        release = mat_data['RELEASE'][0, 0]
        annolist = release['annolist'][0]
        # The 'act' array in RELEASE corresponds by index to 'annolist'
        activity_data_from_mat = release['act']

        # Specific activities and their mapping, as derived from the user's provided script
        # The keys are the exact activity names to look for in the MAT file.
        # The values are the target class names ('sitting', 'standing', 'walking').
        specific_activity_to_class_map = {
            "sitting, talking in person, on the phone, computer, or text messaging, light effort": "sitting",
            "sitting quietly": "sitting",
            "walking, for exerceise, with ski poles": "walking",
            # Using exact string from user's script, including typo
            "paddle boarding, standing": "standing",
            "standing, doing work": "standing",
            "skating, ice dancing": "walking"

        }
        # Create a set of the raw activity names for efficient lookup
        target_raw_activities_set = set(specific_activity_to_class_map.keys())

        data_for_dataframe = []
        print("Processing annotations and filtering images by the new specific activity list...")

        images_skipped_missing_name = 0
        images_skipped_file_not_exist = 0

        for idx, image_annotation_entry in enumerate(annolist):
            image_name_str = None
            try:
                # Extract image name from the annolist structure
                if 'image' in image_annotation_entry.dtype.names and \
                        isinstance(image_annotation_entry['image'], np.ndarray) and image_annotation_entry[
                    'image'].size > 0 and \
                        isinstance(image_annotation_entry['image'][0, 0], np.void) and \
                        'name' in image_annotation_entry['image'][0, 0].dtype.names and \
                        isinstance(image_annotation_entry['image'][0, 0]['name'], np.ndarray) and \
                        image_annotation_entry['image'][0, 0]['name'].size > 0:
                    image_name_str = image_annotation_entry['image'][0, 0]['name'][0]
                else:
                    images_skipped_missing_name += 1
                    continue
            except Exception:
                images_skipped_missing_name += 1
                continue

            # Check if the actual image file exists on disk
            full_image_path = self.raw_images_path / image_name_str
            if not full_image_path.exists():
                images_skipped_file_not_exist += 1
                continue

            # Extract the raw activity name for this image using the existing _extract_activity_name method
            raw_activity_name_from_mat = ""  # Default to empty string
            try:
                if idx < activity_data_from_mat.shape[0]:
                    # Get the activity structure for the current image index
                    current_activity_struct = activity_data_from_mat[idx, 0]
                    raw_activity_name_from_mat = self._extract_activity_name(
                        current_activity_struct)  # This returns '' if not found
                # else: activity remains "" if no corresponding entry in activity_data_from_mat
            except (IndexError, KeyError, TypeError):
                # In case of unexpected structure or missing data for this index in activity_data_from_mat
                pass  # raw_activity_name_from_mat will remain ""

            # Filter based on whether the extracted raw activity name is in our target set
            if raw_activity_name_from_mat in target_raw_activities_set:
                # If it is, get the mapped class (e.g., "sitting")
                mapped_class_name = specific_activity_to_class_map[raw_activity_name_from_mat]

                data_for_dataframe.append({
                    'image_name': image_name_str,
                    'activity': raw_activity_name_from_mat,  # Store the raw activity name that was matched
                    'class': mapped_class_name
                })

        if images_skipped_missing_name > 0:
            print(
                f"Note: Skipped {images_skipped_missing_name} entries due to missing/malformed image name in annotations.")
        if images_skipped_file_not_exist > 0:
            print(
                f"Note: Skipped {images_skipped_file_not_exist} images because their files were not found in {self.raw_images_path}.")

        if not data_for_dataframe:
            print("No images found matching the activities in the new specific_activity_to_class_map. "
                  "Please check the mapping, your data, and ensure image files exist.")
            self.df = pd.DataFrame()  # Ensure df is an empty DataFrame
            return

        self.df = pd.DataFrame(data_for_dataframe)
        # Drop duplicates based on image_name, keeping the first encountered class assignment for an image.
        # This matches the behavior of both the original class and the user's provided script logic.
        self.df = self.df.drop_duplicates(subset=['image_name'], keep='first')
        print(f"Loaded and filtered {len(self.df)} unique images using the new specific activity list.")

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
        """Map activities to simplified classes using self.CLASS_MAPPING.
        Note: This method is NOT used by the modified load_and_filter_data,
        which uses its own hardcoded mapping based on the user's script.
        It's kept for potential other uses or if the class is later reverted.
        """
        for cls, activities_list in self.CLASS_MAPPING.items():
            if activity in activities_list:  # Assumes 'activity' arg is already normalized (e.g., lowercased)
                return cls
        return None  # Return None if no mapping found

    def create_directory_structure(self):
        """Create processed directory structure for train/val/test splits"""
        if self.df is None or self.df.empty:
            print("DataFrame is empty. Skipping directory structure creation.")
            return

        print(f"Creating directory structure under: {self.processed_path}")
        # Ensure CLASS_DIRS matches the classes produced by the new filtering logic.
        # The current self.CLASS_DIRS = ['standing', 'sitting', 'walking'] is compatible.
        actual_classes_in_df = self.df['class'].unique()
        for class_dir in self.CLASS_DIRS:
            if class_dir not in actual_classes_in_df:
                print(
                    f"Warning: CLASS_DIRS contains '{class_dir}' but it's not found in the filtered data after new mapping.")

        for class_dir in actual_classes_in_df:  # Create dirs only for classes present in the filtered df
            if class_dir not in self.CLASS_DIRS:
                print(
                    f"Warning: Filtered data contains class '{class_dir}' which is not in self.CLASS_DIRS. Directory will be created but might not be expected by other parts of the system if they rely on initial self.CLASS_DIRS.")
            for split in ['train', 'val', 'test']:
                path = self.processed_path / class_dir / split
                path.mkdir(parents=True, exist_ok=True)
        print("Directory structure created for classes present in the filtered data.")

    def split_and_save_dataset(self):
        """Split and save dataset into directories"""
        if self.df is None or self.df.empty or 'class' not in self.df.columns:
            print("DataFrame is not properly loaded or 'class' column is missing. Cannot split.")
            return

        # Ensure there are samples to stratify on, and more than 1 class for stratification
        if self.df['class'].nunique() < 2 and len(self.df) > 1:
            print("Warning: Stratification requires at least 2 classes. Performing non-stratified split.")
            stratify_col = None
        elif self.df['class'].nunique() == 0 and len(self.df) > 0:  # Should be caught by df.empty earlier
            print("Warning: No classes assigned, cannot stratify.")
            stratify_col = None
        elif self.df.empty:
            print("DataFrame is empty. Cannot split.")  # Already handled, but good for clarity
            return
        else:
            stratify_col = self.df['class']

        # Define split ratios directly
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1

        try:
            train_df, temp_df = train_test_split(
                self.df,
                test_size=(val_ratio + test_ratio),
                stratify=stratify_col,
                random_state=42
            )
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(test_ratio / (val_ratio + test_ratio)),
                stratify=temp_df['class'] if stratify_col is not None else None,
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

        self._save_split_images(train_df, 'train')
        self._save_split_images(val_df, 'val')
        self._save_split_images(test_df, 'test')

        print("\n🎉 Dataset organization completed!")
        print(f"📂 Total unique images processed for defined classes: {len(self.df)}")
        print(f"📂 Final directory structure for splits:")
        actual_classes_for_print = sorted(list(self.df['class'].unique()))
        for class_name in actual_classes_for_print:
            print(f"  - {class_name.capitalize()}:")
            print(f"    - Training:   {self.processed_path / class_name / 'train'}")
            print(f"    - Validation: {self.processed_path / class_name / 'val'}")
            print(f"    - Testing:    {self.processed_path / class_name / 'test'}")
        if not actual_classes_for_print:
            print("  (No classes found in the final DataFrame to create directories for splits)")

    def _save_split_images(self, df_split, split_name):
        """Save images to corresponding split directory"""
        if df_split.empty:
            print(f"\nNo images to save for {split_name} split.")
            return

        saved_count = 0
        # Initialize class_counts based on actual classes in the current DataFrame split
        # or self.CLASS_DIRS if you want to ensure all predefined dirs are reported.
        # Using actual classes present in the split is more accurate for reporting.
        class_counts = {cls: 0 for cls in df_split['class'].unique()}

        print(f"\nCopying images for {split_name} split...")
        for _, row in df_split.iterrows():
            src = self.raw_images_path / row['image_name']
            if 'class' not in row or pd.isna(row['class']):
                print(f"  Warning: Skipping image {row['image_name']} due to missing or invalid class.")
                continue

            dst_dir = self.processed_path / row['class'] / split_name
            # dst_dir.mkdir(parents=True, exist_ok=True) # Already created
            dst = dst_dir / row['image_name']

            if src.exists():
                try:
                    shutil.copy2(src, dst)
                    saved_count += 1
                    if row['class'] in class_counts:
                        class_counts[row['class']] += 1
                    # else: # This case should not happen if class_counts is initialized from df_split['class'].unique()
                    #    print(f"  Warning: Image {row['image_name']} has class '{row['class']}' not initialized in counts for {split_name} split.")
                except Exception as e:
                    print(f"  Error copying {src} to {dst}: {e}")
            else:
                print(f"  Warning: Source image not found, cannot copy: {src}")

        if saved_count > 0:
            print(f"✅ Successfully saved {saved_count} images to {split_name} split.")
            # The print statement below needs to be generic if classes change
            # print(f"📁 Location: {self.processed_path / '[class]' / split_name}".replace('[class]', '{class}'))
            print(f"📁 Target location pattern: {self.processed_path / '<class_name>' / split_name}")
            print("📊 Class distribution for this split:")
            for cls, count in sorted(class_counts.items()):  # Sort for consistent output
                if count > 0:
                    print(f"  - {str(cls).capitalize()}: {count} images")
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

        plt.figure(figsize=(8, 6))
        counts.plot(kind='bar',
                    color=['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD'])  # More colors if needed
        plt.title('Class Distribution of Filtered MPII Poses (New Filter)')
        plt.xlabel('Pose Class')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45, ha="right")  # Rotate if class names are long
        plt.tight_layout()

        plot_filename = "class_distribution_new_filter.png"
        save_path = self.plots_path / plot_filename
        try:
            plt.savefig(save_path)
            print(f"\n📊 Class distribution plot saved to: {save_path}")
            plt.close()
        except Exception as e:
            print(f"Error saving class distribution plot: {e}")
            plt.show()

    def plot_class_examples(self, num_examples=1):
        """Display example images for each class and save the plot"""
        if self.df is None or self.df.empty or 'class' not in self.df.columns:
            print("DataFrame is not loaded or 'class' column is missing. Cannot plot examples.")
            return

        unique_classes = sorted(self.df['class'].dropna().unique())
        if not unique_classes:
            print("No classes with data found to plot examples.")
            return

        num_classes = len(unique_classes)
        if num_classes == 0:  # Should be caught by previous check
            print("No unique classes to plot examples for.")
            return

        fig, axs = plt.subplots(num_examples, num_classes, figsize=(5 * num_classes, 5 * num_examples), squeeze=False)

        print("\n🖼️ Generating class example plot (New Filter)...")
        for i, cls_name in enumerate(unique_classes):
            class_samples_df = self.df[self.df['class'] == cls_name]
            if class_samples_df.empty:
                for k_ex in range(num_examples):  # For each example row in subplot
                    axs[k_ex, i].set_title(f"{str(cls_name).capitalize()}\n(No Samples)")
                    axs[k_ex, i].axis('off')
                continue

            samples_to_plot = class_samples_df.sample(min(num_examples, len(class_samples_df)), random_state=42)

            for k_ex, (_, sample_row) in enumerate(samples_to_plot.iterrows()):
                img_path = self.raw_images_path / sample_row['image_name']
                ax_current = axs[k_ex, i]
                if img_path.exists():
                    try:
                        img = plt.imread(img_path)
                        ax_current.imshow(img)
                        ax_current.set_title(f"{str(cls_name).capitalize()} (Example {k_ex + 1})")
                    except Exception as e_img:
                        ax_current.set_title(f"{str(cls_name).capitalize()}\nError loading:\n{img_path.name}",
                                             fontsize=8)
                        print(f"  Error loading image {img_path}: {e_img}")
                else:
                    ax_current.set_title(f"{str(cls_name).capitalize()}\nNot found:\n{img_path.name}", fontsize=8)
                    print(f"  Warning: Example image not found: {img_path}")
                ax_current.axis('off')

        # Remove empty subplots if num_examples > actual examples for some classes
        for i in range(num_classes):
            class_samples_df = self.df[self.df['class'] == unique_classes[i]]
            actual_samples_for_class = len(class_samples_df)
            for k_ex in range(min(num_examples, actual_samples_for_class), num_examples):
                # This example was not plotted because there were not enough samples
                # but the subplot axes exist, turn them off if they weren't already.
                if k_ex < axs.shape[0] and i < axs.shape[1] and axs[
                    k_ex, i].get_title() == "":  # Only if not already handled
                    axs[k_ex, i].axis('off')

        plt.tight_layout()
        plot_filename = "class_examples_new_filter.png"
        save_path = self.plots_path / plot_filename
        try:
            plt.savefig(save_path)
            print(f"🖼️ Class examples plot saved to: {save_path}")
            plt.close()
        except Exception as e:
            print(f"Error saving class examples plot: {e}")
            plt.show()


if __name__ == "__main__":
    try:
        print("Initializing Pose Dataset Processor...")
        # Ensure this path points to the directory containing 'images' and the .mat file
        processor = PoseDatasetProcessor(base_path='../data')

        processor.load_and_filter_data()

        if processor.df is not None and not processor.df.empty:
            processor.create_directory_structure()
            processor.split_and_save_dataset()

            processor.plot_class_distribution()
            processor.plot_class_examples(num_examples=2)  # Plot 2 examples per class
        else:
            print("No data loaded or filtered. Exiting.")

    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR - FILE NOT FOUND: {e}")
    except NotADirectoryError as e:
        print(f"\nCRITICAL ERROR - DIRECTORY NOT FOUND: {e}")
    except Exception as e:
        import traceback

        print(f"\nAN UNEXPECTED ERROR OCCURRED: {str(e)}")
        print("Traceback:")
        traceback.print_exc()