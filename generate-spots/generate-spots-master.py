from diffractio import plt, np, um, mm, nm, degrees
from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
from diffractio.utils_drawing import draw_several_fields
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
from skimage import measure
from PIL import Image

""" For Linux install """
import matplotlib.font_manager as fm  # Fixing font

fe = fm.FontEntry(
    fname="/usr/share/fonts/cm-unicode/cmunrm.otf",  # Path to custom installed font (in this case computer modern roman)
    name="cmunrm-manual",
)
fm.fontManager.ttflist.insert(0, fe)  # or append is fine
plt.rcParams["font.family"] = fe.name
plt.rcParams["font.size"] = 24
plt.rcParams["mathtext.fontset"] = "cm"
##### END PREAMBLE #####

cplot = "gray"

original_shape_directory = "to-analyze"
padded_directory = "padded"
render_directory = "renders"

saveformat = "png"

#Adds a blockage that can be configured to be alike what is used in experimental setup
blockage = False

# This blockage is the size of the rerendered image
blockage_width = 1.5 * mm

# For simulations
res = 2048
extent = 5 * mm  # note this is +/- i.e. will get doubled
wavelength = 532 * nm
beam_width = 1.2 * blockage_width  

simplified_model_res = 100 # 250 is used in study

x0 = np.linspace(-extent, extent, res)
y0 = np.linspace(-extent, extent, res)

def padding_image(image, lx, ly, window_real_size, image_real_size):
    x_resized = int(lx * image_real_size / window_real_size)
    y_resized = int(ly * image_real_size / window_real_size)
    image_array = np.asarray(
        Image.open(image).resize((x_resized, y_resized), Image.Resampling.LANCZOS)
    )
    XY_padding = np.ones([lx, ly, 4], dtype=int) * int(255)
    x_centering = (lx - image_array.shape[0]) // 2  # floor division
    y_centering = (ly - image_array.shape[1]) // 2

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            if len(image_array[i, j]) == 3:
                XY_padding[x_centering + i, y_centering + j] = np.append(
                    image_array[i, j], [255]
                )

            if len(image_array[i, j]) == 4:
                XY_padding[x_centering + i, y_centering + j] = image_array[i, j]

                if image_array[i, j][3] == 0:
                    XY_padding[x_centering + i, y_centering + j] = [
                        255,
                        255,
                        255,
                        255,
                    ]
                    print("danger", image_array[i, j])

    return Image.fromarray(XY_padding.astype("uint8"))

def calc_predicted_dist(image, window_real_size, working_res):
    x_resized = working_res
    y_resized = working_res
    image_array = np.asarray(
        Image.open(os.path.join(original_shape_directory,image))
        .convert("L")
        .resize((x_resized-2, y_resized-2), Image.Resampling.NEAREST)
    )
    image=image.replace(".png", "")

    # Pad with one white pixel in each direction to help contour finder
    image_array = np.hstack((image_array, 255*np.ones((y_resized-2,1))))
    image_array = np.hstack((255*np.ones((y_resized-2,1)),image_array))
    image_array = np.vstack((image_array, 255*np.ones((1,x_resized))))
    image_array = np.vstack((255*np.ones((1,x_resized)),image_array))
    
    interior_coords = np.argwhere(image_array == 0)
    outline = measure.find_contours(image_array, 1)
    outline_coords = outline[0]
    # In case detection splits up path
    for i in range(1,len(outline)):
        outline_coords = np.concatenate((outline_coords , outline[i]), axis=0)

    fig, ax = plt.subplots()
    ax.imshow(image_array, cmap=plt.cm.gray)
    ax.plot(outline_coords[:, 1], outline_coords[:, 0], linewidth=2)
    plt.savefig(
        os.path.join(render_directory, f"contour_{image}.{saveformat}"),
        format=saveformat,
        bbox_inches="tight",
        dpi=800,
    ) 
    
    rescale_length =(window_real_size / working_res) 
    convert_to_phase =  2 * np.pi / wavelength 

    nn = len(interior_coords)
    N = len(outline_coords)

    res = np.zeros((working_res, working_res))
    counter = 0
    
    distArr = []

    for intp in interior_coords:
        print(f"Calculating simplified model for {image}: {100*counter/nn:.2f}%")
        counter += 1
        cosCumSum = 0
        sinCumSum = 0
        for outp in outline_coords:
            dist = round(rescale_length * np.sqrt((intp[0] - outp[0]) ** 2 + (intp[1]-outp[1]) ** 2),-1) # Round to nearest 0.1um
            phase = convert_to_phase * dist
            distArr += [dist]
            cosCumSum += np.cos(phase)
            sinCumSum += np.sin(phase)
        res[intp[0],intp[1]] = np.sqrt(cosCumSum**2 + sinCumSum**2) / N
    plt.clf()

    np.savetxt(os.path.join(render_directory, "prediction_" +image + ".txt"), res, delimiter="\t")

    fig = plt.figure()
    pixplot = plt.imshow(res)
    ax = plt.gca()
    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    cbar = plt.colorbar(pixplot, cax=cax)
    cbar.set_label("Intensity")
    plt.savefig(
        os.path.join(render_directory, image + f".{saveformat}"),
        format=saveformat,
        bbox_inches="tight",
        dpi=800,
    )
    mpl.image.imsave(
        os.path.join(
            render_directory,
            cplot + "_" + image + "_predicted." + saveformat,
        ),
        res,
        cmap=cplot,
    )

def execute_predict_images(real_size , resolution):
    directory_original = os.fsencode(original_shape_directory)
    for file in os.listdir(directory_original):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            print(filename)
            calc_predicted_dist(filename, real_size, resolution)
            plt.close()

def padding_redrawing():
    directory_original = os.fsencode(original_shape_directory)
    for file in os.listdir(directory_original):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            (
                padding_image(
                    os.path.join(original_shape_directory, filename),
                    res,
                    res,
                    2 * extent,
                    blockage_width,
                )
            ).save(os.path.join(padded_directory, filename))

def execute_images():
    distance_array = [
        #50 * mm,
        100 * mm,
        #150 * mm,
        #200 * mm,
        #250 * mm,
        500 * mm,
        #2000 * mm,
        #2500 * mm,
        #3000 * mm,
        #3500 * mm,
        10000 * mm
    ]
    directory_padded = os.fsencode(padded_directory)
    for file in os.listdir(directory_padded):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            u0 = Scalar_source_XY(x0, y0, wavelength)
            u0.gauss_beam(r0=(0, 0), w0=beam_width, z0=0, A=1, theta=0.0)

            obstacle = Scalar_mask_XY(x0, y0)

            obstacle.image(
                filename="./" + os.path.join(padded_directory, filename),
                normalize=True,
                canal=0,
                lengthImage=False,
                angle=0 * degrees,
            )

            if blockage:
                obstacle = add_blockage(obstacle)

            for current_distance in distance_array:
                export_filename = f"{filename}_{(current_distance / 10e5):05.2f}m"
                print(os.path.join(padded_directory, export_filename))
                u1 = u0 * obstacle
                arago_point = u1.RS(z=current_distance)

                draw_several_fields(
                    (u1, arago_point), titles=("mask", "propagated"), logarithm=True
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(render_directory, export_filename).replace(
                        ".png","") + f".{saveformat}",
                    format=saveformat,
                    dpi=800,
                    bbox_inches="tight",
                )
                render_images(arago_point.u, export_filename.replace(".png", ""))
                plt.close()

def render_images(u, filename):
    intensity = np.abs(u) ** 2

    mpl.image.imsave(
        os.path.join(
            render_directory,
            f"log_{filename}_propagated_minVal_{np.min(np.log(intensity)):.2f}_maxVal_{np.max(np.log(intensity)):.2f}.{saveformat}",
        ),
        np.log(intensity),
    )
    plt.clf()
    mpl.image.imsave(
        os.path.join(
            render_directory,
            f"gray_{filename}_propagated_minVal_{np.min(intensity):.2f}_maxVal_{np.max(intensity):.2f}.{saveformat}",
        ),
        intensity,
        cmap=cplot,
    )


# Main program starts here

padding_redrawing() # Run only if paddings have to be redrawn
execute_images()
execute_predict_images(blockage_width, simplified_model_res) 
