# Standard library imports
import os
import time
import glob
from datetime import datetime
from ftplib import FTP

# Third-party library imports
from tqdm import tqdm
import numpy as np
import xarray as xr
import netCDF4
from pwinput import pwinput
from landsatxplore.earthexplorer import EarthExplorer


class landsat8_processor:
    def __init__(self, path: str, row: str, directory: str) -> None:
        """
        Initializes the Landsat 8 processing routine for downloading and processing GoLIVE and Earth Explorer data.

        Sets up paths and directories for storing downloaded files and processed data. Initializes attributes
        for handling GoLIVE filenames, Landsat 8 identifiers, and image dimensions.

        Attributes:
            path (str): The Landsat path identifier.
            row (str): The Landsat row identifier.
            path_row (str): Combination of path and row identifiers, formatted as 'p{path}_r{row}'.
            golive_filenames (list): List of filenames of GoLIVE NetCDF4 files to be processed.
            golive_image_dictionary (dict): Dictionary mapping date strings to lists of GoLIVE filenames containing those dates.
            landsat8_identifiers (set): Set of Landsat 8 image identifiers derived from GoLIVE files.
            landsat8_images (dict): Dictionary mapping acquisition dates to filenames of downloaded Landsat 8 images.
            xy_boundaries (list): List of tuples representing the minimum and maximum x and y coordinates for image boundaries.
            image_height (int): Height of the image in pixels.
            image_width (int): Width of the image in pixels.
            image_depth (int): Number of GoLIVE image files.
            directory (str): Base directory for storing downloaded and processed files.
            golive_directory (str): Subdirectory within `directory` for GoLIVE files.
            earth_explorer_directory (str): Subdirectory within `directory` for Earth Explorer files.
        """
        # Initialize attributes
        self.path = path
        self.row = row
        self.path_row = f"p{self.path}_r{self.row}"
        self.golive_filenames = []
        self.golive_image_dictionary = {}
        self.landsat8_identifiers = set()
        self.landsat8_images = {}
        self.xy_boundaries = [(np.inf, -np.inf), (np.inf, -np.inf)]
        self.image_height, self.image_width, self.image_depth = 0, 0, 0

        # Set up directory paths
        self.directory = os.path.join(directory, self.path_row)
        self.golive_directory = os.path.join(self.directory, "golive")
        self.earth_explorer_directory = os.path.join(self.directory, "earthexplorer")

        # Create directories if they do not exist
        for dir_path in [
            self.directory,
            self.golive_directory,
            self.earth_explorer_directory,
        ]:
            os.makedirs(dir_path, exist_ok=True)

    def golive_download(self) -> list:
        """
        Downloads GoLIVE NetCDF4 (.nc) files for the specified path and row from the FTP server.

        This method connects to the FTP server, retrieves a list of available NetCDF4 files,
        and downloads those that have not been previously downloaded. The files are saved to
        the specified directory, and the list of GoLIVE filenames is updated. It also maintains
        a record of successfully downloaded files.

        GoLIVE data provided by https://nsidc.org/data/nsidc-0710/versions/1.

        Returns:
            downloaded_files (list): A list of filenames for the files that were successfully downloaded during this function call.
        """
        downloaded_files = (
            []
        )  # List to track the filenames of successfully downloaded files

        try:
            # Connect to the FTP server and navigate to the target directory
            with FTP("dtn.rc.colorado.edu") as golive_FTP:
                golive_FTP.login(user="anonymous", passwd="")
                golive_directory = f"/work/nsidc0710/nsidc0710_landsat8_golive_ice_velocity_v1.1/{self.path_row}"
                golive_FTP.cwd(golive_directory)

                # Retrieve and filter the list of files from the FTP server
                all_files = golive_FTP.nlst()
                self.golive_filenames = [f for f in all_files if f.endswith(".nc")]

                # Identify files that have not yet been downloaded
                existing_files = set(
                    glob.glob(os.path.join(self.golive_directory, "*.nc"))
                )
                files_to_download = [
                    f
                    for f in self.golive_filenames
                    if os.path.join(self.golive_directory, f) not in existing_files
                ]

                # Ensure the download directory exists
                os.makedirs(self.golive_directory, exist_ok=True)

                # Download the files
                if files_to_download:
                    for file_name in tqdm(
                        files_to_download,
                        desc=f"{time.strftime('%H:%M:%S')} Downloading files for {self.path_row}",
                        unit="file",
                    ):
                        local_file_path = os.path.join(self.golive_directory, file_name)
                        try:
                            with open(local_file_path, "wb") as file:
                                golive_FTP.retrbinary(f"RETR {file_name}", file.write)
                            downloaded_files.append(
                                file_name
                            )  # Record successful downloads
                        except Exception as download_error:
                            print(f"Error downloading {file_name}: {download_error}")

        except Exception as ftp_error:
            print(f"Error connecting to FTP server or processing files: {ftp_error}")

        finally:
            # Update the dictionary with the date information extracted from filenames
            year_dayofyear = set()
            for filename in self.golive_filenames:
                parts = filename.split("_")
                if len(parts) >= 8:
                    year_dayofyear.update(
                        [f"{parts[4]}_{parts[5]}", f"{parts[6]}_{parts[7]}"]
                    )

            self.golive_image_dictionary = {
                date: [f for f in self.golive_filenames if date in f]
                for date in year_dayofyear
            }

        return downloaded_files

    def earth_explorer_download(self) -> list:
        """
        Logs in to the USGS Earth Explorer and downloads Landsat 8 imagery based on the images
        listed in the NetCDF4 files from GoLIVE. The images are saved in a directory structured
        by acquisition date, and the paths to the downloaded Landsat 8 images are stored in
        self.landsat8_images.

        Landsat 8 imagery provided by https://earthexplorer.usgs.gov/.

        This method performs the following steps:
        1. Logs in to Earth Explorer using provided credentials.
        2. Retrieves a list of GoLIVE files if not already available.
        3. Extracts Landsat 8 image identifiers from the GoLIVE files.
        4. Checks for and skips already downloaded images.
        5. Attempts to download each required Landsat 8 image and saves it in the specified directory.
        6. Logs out of Earth Explorer session upon completion.

        Returns:
            list: A list of file paths for the Landsat 8 images that were successfully downloaded.
        """

        # Prompt for Earth Explorer credentials and log in
        while True:
            ee_username = input("Earth Explorer Username: ")
            ee_password = pwinput("Earth Explorer Password: ")

            try:
                self.earth_explorer = EarthExplorer(ee_username, ee_password)
                break
            except Exception as login_error:
                print(f"{login_error} Incorrect username or password.")

        # Retrieve GoLIVE filenames if not already set
        if not self.golive_filenames:
            self.golive_filenames = glob.glob(
                os.path.join(self.golive_directory, "*.nc")
            )

        # Build a set of Landsat 8 identifiers from GoLIVE files
        landsat_identifiers = set()
        for nc_filename in tqdm(
            self.golive_filenames,
            desc=f"{time.strftime('%H:%M:%S')} Creating Landsat 8 product list for {self.path_row}",
        ):
            full_nc_path = os.path.join(self.golive_directory, nc_filename)
            try:
                with netCDF4.Dataset(
                    full_nc_path, "r", format="NETCDF4"
                ) as golive_info:
                    image_filenames = [
                        golive_info["input_image_details"].getncattr("image1_filename"),
                        golive_info["input_image_details"].getncattr("image2_filename"),
                    ]
                    for filename in image_filenames:
                        if "T2" in filename or "RT" in filename:
                            image_datetime = datetime.strptime(
                                filename.split("_")[3], "%Y%m%d"
                            )
                            image_date = f"{image_datetime.year}{image_datetime.timetuple().tm_yday:03d}"
                            updated_filename = (
                                f"LC8{self.path}{self.row}{image_date}LGN00"
                            )
                            landsat_identifiers.add(updated_filename)
                        else:
                            landsat_identifiers.add(filename.split("_")[0])
            except Exception as netcdf4_error:
                print(f"Error processing {nc_filename}: {netcdf4_error}")

        # Check for already downloaded files
        downloaded_files = os.listdir(self.earth_explorer_directory)
        existing_image_dates = {
            f"{datetime.strptime(f.split('_')[3], '%Y%m%d').year}{datetime.strptime(f.split('_')[3], '%Y%m%d').timetuple().tm_yday:03d}": f
            for f in downloaded_files
        }

        # Track downloaded files
        successfully_downloaded_files = []

        # Download necessary Landsat 8 files
        for landsat8_data in landsat_identifiers:
            image_date = landsat8_data[9:16]
            if image_date in existing_image_dates:
                self.landsat8_images[image_date] = existing_image_dates[image_date]
                successfully_downloaded_files.append(existing_image_dates[image_date])
                continue

            for file_version in range(5):
                LGN_version = f"LGN{file_version:02}"
                filename = f"{landsat8_data[:-5]}{LGN_version}"
                try:
                    print(f"Trying {filename}")

                    # Sometimes files partially download and assume complete download. Check filesize and retry files that are below a size threshold.
                    while True:
                        download_filepath = self.earth_explorer.download(
                            filename, output_dir=self.earth_explorer_directory
                        )
                        if os.path.getsize(download_filepath) > 512000000:
                            break
                        else:
                            os.remove(download_filepath)

                    print(
                        f"{time.strftime('%H:%M:%S')} {filename} Download Complete at {download_filepath}"
                    )
                    self.landsat8_images[image_date] = os.path.basename(
                        download_filepath
                    )
                    successfully_downloaded_files.append(download_filepath)
                    break
                except Exception:
                    continue
            else:
                print(f"{time.strftime('%H:%M:%S')} {landsat8_data} Download Failed")

        # Log out of the Earth Explorer session
        self.earth_explorer.logout()

        return successfully_downloaded_files

    def prepare_golive_dimensions(self) -> None:
        """
        Computes the boundaries, height, and width for the full extent of the GoLIVE image set.

        This method iterates through all GoLIVE image files to determine the spatial extent
        and calculates the dimensions of the images based on a fixed pixel size (300x300).

        Attributes:
            self.xy_boundaries (list of tuples): List containing the minimum and maximum x and y coordinates.
            self.image_height (int): Height of the image in pixels.
            self.image_width (int): Width of the image in pixels.
            self.image_depth (int): Number of GoLIVE image files.
        """

        # Initialize boundaries to extreme values
        x_min, x_max = np.inf, -np.inf
        y_min, y_max = np.inf, -np.inf

        # Process each GoLIVE file to determine the spatial boundaries
        for filename in tqdm(
            self.golive_filenames,
            desc=f"{time.strftime('%H:%M:%S')} Calculating image boundaries for {self.path_row}",
        ):
            with xr.open_dataset(
                os.path.join(self.golive_directory, filename)
            ) as golive_data:
                x_bounds = (np.min(golive_data.x.values), np.max(golive_data.x.values))
                y_bounds = (np.min(golive_data.y.values), np.max(golive_data.y.values))

                # Update the global boundaries
                x_min, x_max = min(x_min, x_bounds[0]), max(x_max, x_bounds[1])
                y_min, y_max = min(y_min, y_bounds[0]), max(y_max, y_bounds[1])

        # Set the boundaries and calculate image dimensions
        self.xy_boundaries = [(x_min, x_max), (y_min, y_max)]
        self.image_height = int(((y_max - y_min) // 300) + 1)
        self.image_width = int(((x_max - x_min) // 300) + 1)
        self.image_depth = len(self.golive_filenames)
