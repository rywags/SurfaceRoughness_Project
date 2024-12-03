# Standard library imports
import os
import glob
from datetime import datetime, timedelta
import tarfile
import tempfile
import shutil

# Third-party library imports
from tqdm import tqdm
import numpy as np
import xarray as xr
import rasterio
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import binary_dilation, median_filter

# Custom library imports
import surfaceroughness_downloads as sr_d


class image_processor:
    def __init__(
        self,
        image_date: str,
        landsat8_process: sr_d.landsat8_processor,
    ) -> None:
        """
        Initializes the Landsat 8 data for a specific date by unpacking and processing the Band 8 image.

        Attributes:
            image_date (str): The date of the Landsat 8 image in 'YYYYDDD' format.
            image_date_nc (str): Formatted date for use in file names, in 'YYYY_DDD' format.
            landsat8_process (landsat8_processor): An instance of the Landsat8 processor class that contains
                                                    information and methods for processing Landsat 8 data.
            low_correlation_threshold (float): The threshold for filtering correlation images to determine what should be considered a cloud or not.
            earth_explorer_file_location (str): Path to the Earth Explorer file containing the Band 8 image.
            output_directory (str): Directory path for saving output files, specific to the image date.
            temp_directory (str): Path to a temporary directory for extracting and processing the Band 8 image.
            band_8_filepath (str): Path to the extracted Band 8 image file.
            band_8_data (numpy.ndarray): Array containing the data of the Band 8 image.
            meta (dict): Metadata of the Band 8 image including its data type, count, and other properties.
        """
        self.image_date = image_date
        self.image_date_nc = f"{image_date[:4]}_{image_date[4:]}"
        self.landsat8_process = landsat8_process
        self.low_correlation_threshold = 0.0
        # Cache the path to the Earth Explorer file
        self.earth_explorer_file_location = os.path.join(
            landsat8_process.earth_explorer_directory,
            landsat8_process.landsat8_images.get(image_date),
        )
        # Create and cache the output directory path
        self.output_directory = os.path.join(landsat8_process.directory, image_date)
        os.makedirs(self.output_directory, exist_ok=True)

        # Create a temporary directory and extract Band 8 image
        self.temp_directory = tempfile.mkdtemp()  # Create a temporary directory
        try:
            with tarfile.open(self.earth_explorer_file_location, "r") as tar:
                # Find and extract the Band 8 image
                band_8_member = next(
                    (member for member in tar.getmembers() if "B8.TIF" in member.name),
                    None,
                )
                if band_8_member:
                    tar.extract(
                        band_8_member, self.temp_directory, filter="fully_trusted"
                    )
                    self.band_8_filepath = os.path.join(
                        self.temp_directory, band_8_member.name
                    )
                else:
                    raise FileNotFoundError("Band 8 file not found in the archive.")
            # Load Band 8 image data
            with rasterio.open(self.band_8_filepath) as src:
                self.band_8_data = src.read(1)  # Read the first band
                self.meta = src.meta.copy()  # Copy metadata
                self.corr_meta = self.meta.copy()
                self.meta.update(dtype=rasterio.int16, count=1, nodata=None)
                self.corr_meta.update(dtype=rasterio.float32, count=1, nodata=0)
        except (tarfile.TarError, FileNotFoundError) as error:
            raise RuntimeError(
                f"Error unpacking or processing {self.earth_explorer_file_location}: {error}"
            )

    def __del__(self) -> None:
        """
        Clean up temporary files and directories.

        Used to remove the `self.temp_directory` used to store `self.band_8_filepath`.

        Raises:
            OSError: If the temporary directory cannot be deleted (e.g., it is restricted).
        """
        try:
            if os.path.isdir(self.temp_directory):
                # Initialize progress bar for the cleanup process

                shutil.rmtree(self.temp_directory)
        except OSError as e:
            # Handle exceptions and update the progress bar if provided
            print(f"Error deleting temporary directory {self.temp_directory}: {e}")

    def copy_original_band8(self) -> None:
        """
        Copies the original Band 8 image to the output directory, overwriting if it already exists.

        This function checks if the Band 8 image file is available. It copies the file from its
        source location to the specified output directory. If the file already exists, it will be
        overwritten to handle potential corruption issues.

        Raises:
            FileNotFoundError: If the Band 8 image file is not available.
        """
        if not self.band_8_filepath:
            raise FileNotFoundError("Band 8 image file is not available.")
        # Define the source and destination paths
        source_path = self.band_8_filepath
        destination_path = os.path.join(
            self.output_directory, os.path.basename(source_path)
        )
        try:
            # Copy the file, overwriting if it already exists
            shutil.copy(source_path, destination_path)
        except Exception as e:
            raise RuntimeError(f"Error copying Band 8 image: {e}")

    def create_filtered_image(self) -> str:
        """
        Applies high and low pass filters to the Band 8 image and saves the processed image.

        Filters applied:
        1. High pass filter with a 5x5 kernel.
        2. Absolute value of the high pass filtered image.
        3. Low pass filter with a 17x17 kernel.

        Returns:
            output_path (str): Path to the saved filtered image.

        Raises:
            FileNotFoundError: If the Band 8 file is missing.
            RuntimeError: If an error occurs during filtering or saving.
        """
        if not self.band_8_filepath:
            raise FileNotFoundError("Band 8 file is not available.")
        try:
            # Apply high pass filter with a 5x5 kernel
            low_pass_kernel_1 = np.ones((5, 5)) / 25
            low_pass_image_1 = fftconvolve(
                self.band_8_data, low_pass_kernel_1, mode="same"
            )
            high_pass_image = self.band_8_data - low_pass_image_1
            # Compute the absolute value of the high pass filtered image
            absolute_value_image = np.abs(high_pass_image)
            # Apply low pass filter with a 17x17 kernel
            low_pass_kernel_2 = np.ones((17, 17)) / (17 * 17)
            low_pass_image_2 = fftconvolve(
                absolute_value_image, low_pass_kernel_2, mode="same"
            )
            # Define output file path
            output_path = os.path.join(
                self.output_directory, f"filtered_image_{self.image_date}.TIF"
            )
            # Save the processed image
            with rasterio.open(output_path, "w", **self.meta) as dst:
                dst.write(low_pass_image_2.astype(rasterio.int16), 1)
            return output_path
        except Exception as error:
            raise RuntimeError(f"Error during filtering and saving: {error}")

    def create_correlation_image(
        self,
        correlation_threshold: float = 0.0,
        time_range: int = np.iinfo(np.int16).max,
    ) -> str:
        """
        Creates a correlation image from GoLIVE data, applying a threshold to exclude low-correlation values.

        Aggregates correlation data from multiple GoLIVE files, filters based on a time range and correlation
        threshold, and computes the mean correlation. The resulting correlation image is saved as a TIFF file.

        Parameters:
            correlation_threshold (float): Minimum correlation value to include. Values below this threshold
                                            are set to NaN.
            time_range (int): Maximum allowable difference in days between the target image date and the dates
                            in the GoLIVE files for inclusion in the correlation image.

        Returns:
            output_path (str): File path to the saved correlation image.

        Raises:
            RuntimeError: If an error occurs during the creation of the correlation image.
        """
        try:
            # Format the image date and retrieve associated GoLIVE filenames
            formatted_image_date = f"{self.image_date[:4]}_{self.image_date[4:]}"
            golive_filenames = self.landsat8_process.golive_image_dictionary.get(
                formatted_image_date, []
            )
            # Convert image date to datetime object for filtering
            datetime_image_date = datetime.strptime(formatted_image_date, "%Y_%j")
            # Filter filenames based on time range
            filtered_filenames = [
                filename
                for filename in golive_filenames
                if any(
                    abs(
                        (
                            datetime_image_date
                            - datetime.strptime(
                                f"{filename.split('_')[i]}_{filename.split('_')[i+1]}",
                                "%Y_%j",
                            )
                        ).days
                    )
                    <= time_range
                    for i in [4, 6]
                )
            ]
            # Initialize the correlation cube
            correlation_cube = np.zeros(
                (
                    self.landsat8_process.image_height,
                    self.landsat8_process.image_width,
                    len(filtered_filenames),
                ),
                dtype=np.float32,
            )
            for index, filename in enumerate(filtered_filenames):
                # Load dataset and apply correlation threshold
                golive_dataset = xr.load_dataset(
                    os.path.join(self.landsat8_process.golive_directory, filename)
                )
                corr_data = golive_dataset.corr.values
                corr_data[corr_data < correlation_threshold] = np.nan
                # Compute offsets for placing data into the cube
                j_offset = int(
                    (
                        np.min(golive_dataset.y.values)
                        - self.landsat8_process.xy_boundaries[1][0]
                    )
                    // 300
                )
                i_offset = int(
                    (
                        np.min(golive_dataset.x.values)
                        - self.landsat8_process.xy_boundaries[0][0]
                    )
                    // 300
                )
                # Update the correlation cube with the current file's data
                correlation_cube[
                    j_offset : j_offset + corr_data.shape[0],
                    i_offset : i_offset + corr_data.shape[1],
                    index,
                ] += corr_data
                golive_dataset.close()
            correlation_max_allowance = np.max(correlation_cube, axis=2) / 1.67
            correlation_std_allowance = np.std(correlation_max_allowance) * 1.67
            correlation_cube[
                correlation_cube < correlation_max_allowance[..., np.newaxis]
            ] = np.nan
            correlation_cube[correlation_cube < correlation_std_allowance] = np.nan
            # Compute the mean correlation across all files
            correlation_raster = np.nanmean(correlation_cube, axis=2)
            self.low_correlation_threshold = np.nanmean(correlation_raster) * 0.80
            correlation_raster = np.nan_to_num(correlation_raster)
            # Define and save the output correlation image
            output_path = os.path.join(
                self.output_directory, f"correlation_image_{self.image_date}.TIF"
            )
            with rasterio.open(output_path, "w", **self.corr_meta) as dst:
                dst.write(correlation_raster.astype(rasterio.float32), 1)
            return output_path
        except Exception as error:
            raise RuntimeError(f"Error creating correlation image: {error}")


def cross_plot_landsat_images(
    filtered_image_path: str,
    correlation_image_path: str,
    image_date: str,
    output_plot_path: str,
) -> str:
    """
    DEPRECATED:
        Moving to using R for creation of plots and graphs.

    Creates a cross plot comparing values from filtered and correlation Landsat 8 images.

    This function loads the specified Landsat images, replaces all zero values with NaN
    to exclude them from the plot, and generates a scatter plot showing the relationship
    between the filtered and correlation image values.

    Parameters:
        filtered_image_path (str): File path to the filtered Landsat image (e.g., a .tif file).
        correlation_image_path (str): File path to the correlation Landsat image (e.g., a .tif file).
        image_date (str): Date associated with the images, used for naming the output plot file.
        output_plot_path (str): Directory path where the generated cross plot will be saved.

    Returns:
        plot_filepath (str): Full file path where the cross plot image was saved.
    """

    def load_image(image_path: str) -> np.ndarray:
        """
        Load an image from the specified file path.

        Parameters:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Array containing the image data.
        """
        with rasterio.open(image_path) as src:
            return src.read(1)

    try:
        # Load the filtered and correlation images
        filtered_image = load_image(filtered_image_path)
        correlation_image = load_image(correlation_image_path)
        # Replace zero values with NaN in both images
        filtered_image[filtered_image == 0] = np.nan
        correlation_image[correlation_image == 0] = np.nan
        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(filtered_image.flatten(), correlation_image.flatten(), alpha=0.5)
        plt.xlabel("Filtered Image Values")
        plt.ylabel("Correlation Image Values")
        plt.title(f"Cross Plot for {image_date}")
        plt.grid(True)
        # Save the plot
        plot_filepath = os.path.join(output_plot_path, f"cross_plot_{image_date}.png")
        plt.savefig(plot_filepath)
        plt.close()
        return plot_filepath
    except Exception as error:
        raise RuntimeError(f"Error creating cross plot: {error}")


def create_masked_image(
    correlation_image_filepath: str,
    filtered_image_filepath: str,
    data_information: image_processor,
) -> str:
    """
    Generates a masked image by applying a binary mask to a filtered image, based on a correlation threshold.
    The function also performs noise reduction using median filtering and dilation.

    Parameters:
        correlation_image_filepath (str): Path to the correlation image file.
        filtered_image_filepath (str): Path to the filtered image file.
        data_information (image_processor): Contains metadata and file handling information, including
                                            output directory and image date..

    Returns:
        output_path (str): File path of the saved masked image.
    """
    # Open the correlation and filtered images and load the data
    with rasterio.open(correlation_image_filepath) as correlation_image, rasterio.open(
        filtered_image_filepath
    ) as filtered_image:
        correlation_data = correlation_image.read(1)
        filtered_data = filtered_image.read(1)
    # Create a binary mask based on the correlation threshold
    binary_mask = correlation_data >= data_information.low_correlation_threshold
    # Apply the mask to the filtered data and zero out non-correlated areas
    masked_filtered_data = np.where(binary_mask, filtered_data, 0)
    # Apply median filter for noise reduction
    masked_filtered_data = median_filter(masked_filtered_data, size=8)
    # Dilate areas with zero values to capture cloudy regions
    dilated_zeros = binary_dilation(
        masked_filtered_data == 0, structure=np.ones((17, 17)), iterations=4
    )
    # Mask the dilated zero areas
    masked_filtered_data = np.where(dilated_zeros, 0, masked_filtered_data)
    # Define the output path and save the masked image
    output_path = os.path.join(
        data_information.output_directory,
        f"masked_image_{data_information.image_date}.TIF",
    )
    with rasterio.open(output_path, "w", **data_information.meta) as dst:
        dst.write(masked_filtered_data.astype(rasterio.int16), 1)
    return output_path


def create_average_image(
    image_dates: list,
    data_information: image_processor,
    path_row_information: sr_d.landsat8_processor,
    time_range: int = np.iinfo(np.int16).max,
) -> str:
    """
    Computes the average raster image by selecting and stacking 'masked_image' .TIF files
    within a specified time range, then calculates the average and saves it.

    Parameters:
        image_dates (list): List of image date strings in "%Y_%j" format.
        data_information (image_processor): Contains metadata, output directory, and image date.
        path_row_information (landsat8_processor): Contains directory path information for images.
        time_range (int, optional): Time range in days to include around the base date. Default is max.

    Returns:
        output_path (str): File path of the saved average image.
    """
    # Convert base image date to datetime object
    base_date = datetime.strptime(data_information.image_date, "%Y%j")
    # Define time range boundaries
    start_date = base_date - timedelta(days=time_range)
    end_date = base_date + timedelta(days=time_range)

    def convert_date(date_str):
        """
        Convert image dates to datetime and filter based on the time range
        """
        return datetime.strptime(date_str, "%Y_%j")

    filtered_dates = [
        date for date in image_dates if start_date <= convert_date(date) <= end_date
    ]

    def find_masked_image_files(base_directory: str, date_list: list) -> list:
        """
        Find 'masked_image' .TIF files corresponding to filtered dates
        """
        date_set = set(date.replace("_", "") for date in date_list)
        search_pattern = os.path.join(base_directory, "**", "*masked_image*.TIF")
        return [
            file
            for file in glob.glob(search_pattern, recursive=True)
            if any(date in os.path.basename(file) for date in date_set)
        ]

    masked_image_files = find_masked_image_files(
        base_directory=path_row_information.directory, date_list=filtered_dates
    )
    if not masked_image_files:
        raise ValueError("No masked_image .TIF files found for the given dates.")
    # Initialize variables for stacking
    max_height, max_width = 0, 0
    for file in masked_image_files:
        with rasterio.open(file) as src:
            height, width = src.height, src.width
            max_height = max(max_height, height)
            max_width = max(max_width, width)
    # Create a 3D array for stacking, initialized with NaN
    image_stack = np.full(
        (len(masked_image_files), max_height, max_width), np.nan, dtype=np.float64
    )
    # Read images, apply padding, and stack them
    for i, file in enumerate(masked_image_files):
        with rasterio.open(file) as src:
            data = src.read(1)
            height, width = data.shape
            # Apply padding if necessary
            image_stack[i, :height, :width] = data
    # Compute the average using np.nanmean
    average_image = np.nanmean(image_stack, axis=0)
    # Define the output path
    output_path = os.path.join(
        data_information.output_directory,
        f"average_image_{data_information.image_date}.TIF",
    )
    # Save the average image to the output path
    average_image = np.nan_to_num(average_image, nan=0, posinf=0, neginf=0)
    with rasterio.open(output_path, "w", **data_information.meta) as dst:
        dst.write(average_image.astype(rasterio.int16), 1)
    return output_path
