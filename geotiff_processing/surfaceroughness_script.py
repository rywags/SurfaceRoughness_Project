# Standard library imports
import os
import warnings

# Third-party library imports
from tqdm import tqdm
import matplotlib.pyplot as plt

# Custom library imports
import surfaceroughness_downloads as sr_d
import surfaceroughness_processing as sr_p


def main() -> None:
    """
    Main function to orchestrate the processing of Landsat 8 imagery data.

    Designed to save all outputs to an external drive. Edit the path of `remote_directory` for your own storage location.

    1. Sets up the backend for plotting and disables interactive mode.
    2. Creates the necessary directories for storing data.
    3. Collects user input for Landsat path and row numbers.
    4. Initializes the `landsat8_processor` with the provided inputs.
    5. Downloads GoLIVE and Earth Explorer data and prepares dimensions.
    6. Processes each image date by:
       - Copying the original Band 8 image.
       - Creating filtered and correlation images.
       - Creating an average of all masked images over a certain time frame.
    """

    # Set up the remote directory for storing data
    remote_directory = os.path.abspath(r"E:\Surface_Roughness_Data")
    try:
        os.makedirs(remote_directory, exist_ok=True)
        print(f"Remote directory created at {remote_directory}")
    except Exception as e:
        print(f"Error creating remote directory: {e}")
        return

    # Get user input for Landsat path and row with validation
    path = input("Path (three-digit numeric specifier): ").strip()
    row = input("Row (three-digit numeric specifier): ").strip()

    # Validate that path and row are numeric and exactly three digits long
    if not (path.isdigit() and row.isdigit() and len(path) == 3 and len(row) == 3):
        print(
            "Invalid path or row. Please provide numeric values with exactly three digits."
        )
        return

    # Initialize the Landsat 8 processor
    surface_roughness = sr_d.landsat8_processor(path, row, remote_directory)
    print("Initialized Landsat 8 processor.")

    # Download GoLIVE and Earth Explorer data
    try:
        surface_roughness.golive_download()
        surface_roughness.earth_explorer_download()
        print("Data download complete.")
    except Exception as e:
        print(f"Error during data download: {e}")
        return

    # Prepare dimensions for GoLIVE images
    try:
        surface_roughness.prepare_golive_dimensions()
        print("GoLIVE dimensions prepared.")
    except Exception as e:
        print(f"Error preparing GoLIVE dimensions: {e}")
        return

    # Process each Landsat 8 image date
    for i, image_date in enumerate(surface_roughness.landsat8_images):
        print(
            f"Processing {image_date} ({i} / {len(surface_roughness.landsat8_images)})"
        )
        with tqdm(total=18, desc=f"Initializing {image_date}") as pbar:
            # Initialize the image processor for the current image date
            landsat8image = sr_p.image_processor(
                image_date=image_date, landsat8_process=surface_roughness, pbar=pbar
            )
            try:
                # Copy original Band 8 image
                landsat8image.copy_original_band8()

                # Create filtered and correlation images
                correlation_image = landsat8image.create_correlation_image(
                    correlation_threshold=0.16, time_range=64, pbar=pbar
                )
                filtered_image = landsat8image.create_filtered_image(pbar=pbar)

                # Create masked filtered image
                sr_p.create_masked_image(
                    correlation_image_filepath=correlation_image,
                    filtered_image_filepath=filtered_image,
                    data_information=landsat8image,
                    pbar=pbar,
                )

            except Exception as e:
                print(f"Error processing image date {image_date}: {e}")

            finally:
                del landsat8image

    # Create the averages of the images
    for image_date in tqdm(
        surface_roughness.landsat8_images,
        desc=f"Creating average image set",
    ):
        try:
            landsat8image = sr_p.image_processor(
                image_date=image_date, landsat8_process=surface_roughness
            )
            sr_p.create_average_image(
                image_dates=list(surface_roughness.golive_image_dictionary.keys()),
                data_information=landsat8image,
                path_row_information=surface_roughness,
                time_range=64,
            )
        except Exception as e:
            print(f"Error processing image date {image_date}: {e}")
        finally:
            del landsat8image


if __name__ == "__main__":
    plt.switch_backend("Agg")  # Use non-interactive backend for plotting
    plt.ioff()  # Turn off interactive mode

    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="Mean of empty slice"
    )  # Some rasters are poor quality and all NaN, ignore warning for these

    main()
