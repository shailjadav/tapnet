import time
import cv2
import jax
import jax.numpy as jnp 
import numpy as np
from tapnet.models import tapir_model
from tapnet.utils import model_utils
from ultralytics import SAM
from ultralytics import YOLO
from scipy.spatial import distance
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
from scipy.spatial import procrustes
from scipy.spatial import distance
from sklearn.cluster import KMeans




image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model_depth = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to("cuda")
model_sam = SAM("sam2_b.pt")
model_yolo = YOLO("yolov8x-world.pt") 
model_yolo.set_classes(["cake"])



NUM_POINTS = 11

def load_checkpoint(checkpoint_path):
    ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
    return ckpt_state["params"], ckpt_state["state"]

print("Loading checkpoint...")
params, state = load_checkpoint("tapnet/checkpoints/causal_tapir_checkpoint.npy")

tapir = tapir_model.ParameterizedTAPIR(
    params=params,
    state=state,
    tapir_kwargs=dict(
        use_causal_conv=True, bilinear_interp_with_depthwise_conv=False
    ),
)

def online_model_init(frames, points):
    feature_grids = tapir.get_feature_grids(frames, is_training=False)
    features = tapir.get_query_features(
        frames,
        is_training=False,
        query_points=points,
        feature_grids=feature_grids,
    )
    return features

def online_model_predict(frames, features, causal_context):
    feature_grids = tapir.get_feature_grids(frames, is_training=False)
    trajectories = tapir.estimate_trajectories(
        frames.shape[-3:-1],
        is_training=False,
        feature_grids=feature_grids,
        query_features=features,
        query_points_in_video=None,
        query_chunk_size=64,
        causal_context=causal_context,
        get_causal_context=True,
    )
    causal_context = trajectories["causal_context"]
    del trajectories["causal_context"]
    return {k: v[-1] for k, v in trajectories.items()}, causal_context

def get_frame(video_capture):
    r_val, image = video_capture.read()
    trunc = np.abs(image.shape[1] - image.shape[0]) // 2
    if image.shape[1] > image.shape[0]:
        image = image[:, trunc:-trunc]
    elif image.shape[1] < image.shape[0]:
        image = image[trunc:-trunc]
    return r_val, image


############################################################My Functions##################################################
def add_red_bar_to_frame(frame, value):
    value = max(0, min(value, 20))
    frame_height, frame_width, _ = frame.shape
    bar_height = int((value / 20) * frame_height)
    bar = np.zeros((frame_height, 20, 3), dtype=np.uint8)
    bar[:bar_height, :] = [0, 0, 255]  # Red color in BGR format
    modified_frame = np.hstack((frame, bar))
    return modified_frame


# def create_grid_points(mask, num_points):

#     contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return []

#     # Get the largest contour
#     contour = max(contours, key=cv2.contourArea)

#     # Get bounding box of the contour
#     x, y, w, h = cv2.boundingRect(contour)

#     # Create a grid of candidate points
#     grid_size = int(np.sqrt(num_points * 4))  # Create more candidate points than needed
#     x_step = w / (grid_size - 1)
#     y_step = h / (grid_size - 1)

#     candidate_points = []
#     for i in range(grid_size):
#         for j in range(grid_size):
#             point_x = int(x + i * x_step)
#             point_y = int(y + j * y_step)
#             if 0 <= point_y < mask.shape[0] and 0 <= point_x < mask.shape[1] and mask[point_y, point_x]:
#                 candidate_points.append([point_x, point_y])

#     # If we don't have enough points, return what we have
#     if len(candidate_points) <= num_points:
#         return candidate_points

#     # Select points using a greedy approach
#     selected_points = [candidate_points[0]]  # Start with the first point
#     while len(selected_points) < num_points:
#         best_point = None
#         best_min_distance = 0
#         for point in candidate_points:
#             if point not in selected_points:
#                 min_distance = min(distance.euclidean(point, sp) for sp in selected_points)
#                 if min_distance > best_min_distance:
#                     best_min_distance = min_distance
#                     best_point = point
#         if best_point:
#             selected_points.append(best_point)
#         else:
#             break  
#     return selected_points

def create_grid_points(mask, num_points):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Get all points inside the contour
    y_coords, x_coords = np.where(mask > 0)
    points = np.column_stack((x_coords, y_coords))

    # Use K-means clustering to distribute points evenly
    kmeans = KMeans(n_clusters=num_points, random_state=42)
    kmeans.fit(points)

    # Get the cluster centers as our grid points
    grid_points = kmeans.cluster_centers_.astype(int)

    # Ensure all points are within the mask
    grid_points = [point for point in grid_points if mask[point[1], point[0]] > 0]

    # If we don't have enough points, add more from the contour
    while len(grid_points) < num_points:
        # Find the pair of points with the maximum distance
        max_dist = 0
        max_pair = None
        for i in range(len(grid_points)):
            for j in range(i+1, len(grid_points)):
                dist = distance.euclidean(grid_points[i], grid_points[j])
                if dist > max_dist:
                    max_dist = dist
                    max_pair = (grid_points[i], grid_points[j])
        
        if max_pair:
            # Add a point in the middle of the maximum distance pair
            new_point = ((max_pair[0][0] + max_pair[1][0]) // 2, 
                         (max_pair[0][1] + max_pair[1][1]) // 2)
            if mask[new_point[1], new_point[0]] > 0:
                grid_points.append(new_point)
            else:
                # If the midpoint is not in the mask, find the closest valid point
                valid_points = points[np.random.choice(len(points), 100, replace=False)]
                closest_point = min(valid_points, key=lambda p: distance.euclidean(p, new_point))
                grid_points.append(closest_point)
        else:
            break

    return grid_points


def img_depth(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = image_processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model_depth(**inputs)
        predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(frame.shape[0], frame.shape[1]),
        mode="bicubic",
        align_corners=False,
    )
    output = prediction.squeeze().cpu().numpy()
    output = output - np.min(output)  
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth_cv = np.array(depth)

    return depth_cv


def img_dep_cmb(track_c,depth_cv):
    track_c_d= []
    for i, point in enumerate(track_c):
        x, y = point[:2]
        depth_value = depth_cv[int(y), int(x)]  
        track_c_d.append((x, y, depth_value))
    track_c_d= np.array(track_c_d)
    return track_c_d

def find_optimal_rotation(A, B):
  
    assert A.shape == B.shape, "The point clouds must have the same dimensions and number of points"
    
    # Center the points around the origin
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute the covariance matrix
    H = np.dot(AA.T, BB)

    # SVD to find the rotation
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Ensure a proper rotation matrix with det(R) = 1
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Convert rotation matrix to Euler angles
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    euler_angles = (x, y, z)  # Roll, Pitch, Yaw in radians

    return R, euler_angles 
       

###############################################################################################################################


print("Welcome to the TAPIR live demo.")
print("Please note that if the framerate is low (<~12 fps), TAPIR performance")
print("may degrade and you may need a more powerful GPU.")

print("Creating model...")
online_init_apply = jax.jit(online_model_init)
online_predict_apply = jax.jit(online_model_predict)

print("Initializing camera...")
vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if vc.isOpened():
    rval, frame = get_frame(vc)
else:
    raise ValueError("Unable to open camera.")

query_frame = True
have_point = [False] * NUM_POINTS
query_features = None
causal_state = None
next_query_idx = 0

print("Compiling jax functions (this may take a while...)")
query_points = jnp.zeros([NUM_POINTS, 3], dtype=jnp.float32)
_ = online_init_apply(
    frames=model_utils.preprocess_frames(frame[None, None]),
    points=query_points[None, 0:1],
)
jax.block_until_ready(query_features)
query_features = online_init_apply(
    frames=model_utils.preprocess_frames(frame[None, None]),
    points=query_points[None, :],
)

causal_state = tapir.construct_initial_causal_state(
    NUM_POINTS, len(query_features.resolutions) - 1
)

prediction, causal_state = online_predict_apply(
    frames=model_utils.preprocess_frames(frame[None, None]),
    features=query_features,
    causal_context=causal_state,
)

jax.block_until_ready(prediction["tracks"])

print("Press ESC to exit.")
############################################################My FunctionsVars##################################################
flag=0
grid_point_num=4
f_d_norm=0
selected_points = None
force=0


while rval:
    rval, frame = get_frame(vc)
    if selected_points is None:
        time.sleep(0.1)
    #################################################Center Detection#################################################
        if flag==0:
            results_yolo = model_yolo.predict(frame,conf=0.3)
            centers = []
            for r in results_yolo:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    centers.append([center_x, center_y])
            if centers:
                results_sam = model_sam(frame,points=centers[0], labels=[1])
                for r in results_sam:
                    # Access the pixel coordinates of the first mask
                    if len(r.masks.xy) > 0:
                        points_in_segment = r.masks.xy[0]  # get coordinates of the first mask
                        mask=r.masks.data[0].cpu().numpy()
                        selected_points = create_grid_points(mask, grid_point_num)
                for r in results_sam:
                    im_array = r.plot()  # plot a BGR numpy array of predictions
                    cv2.imshow("SAM Results", im_array)
            
                        
    if selected_points:
        if query_frame:
            for point in selected_points:
                if next_query_idx < NUM_POINTS:
                    y, x = point
                    query_points = jnp.array([0, x, y], dtype=jnp.float32)
                    init_query_features = online_init_apply(
                        frames=model_utils.preprocess_frames(frame[None, None]),
                        points=query_points[None, None],
                    )
                    query_features, causal_state = tapir.update_query_features(
                        query_features=query_features,
                        new_query_features=init_query_features,
                        idx_to_update=np.array([next_query_idx]),
                        causal_state=causal_state,
                    )
                    have_point[next_query_idx] = True
                    next_query_idx = (next_query_idx + 1) % NUM_POINTS
            query_frame = False

        if next_query_idx > 0:
            prediction, causal_state = online_predict_apply(
                frames=model_utils.preprocess_frames(frame[None, None]),
                features=query_features,
                causal_context=causal_state,
            )
            track = prediction["tracks"][0, :, 0]
            occlusion = prediction["occlusion"][0, :, 0]
            expected_dist = prediction["expected_dist"][0, :, 0]
            visibles = model_utils.postprocess_occlusions(occlusion, expected_dist)
            track = np.round(track)

            for i, _ in enumerate(have_point):
                if visibles[i] and have_point[i]:
                    cv2.circle(
                        frame, (int(track[i, 0]), int(track[i, 1])), 5, (255, 0, 0), -1
                    )
                    if track[i, 0] < 16 and track[i, 1] < 16:
                        print((i, next_query_idx))

        if flag==0:
            first_track = track[:grid_point_num]
            first_track=first_track.at[:, 0].set(480 - first_track[:, 0])
            depth_cv= img_depth(frame[:, ::-1])
            first_track_c_d=img_dep_cmb(first_track,depth_cv)
            d_norm_first = np.linalg.norm(distance.squareform(distance.pdist(first_track, 'euclidean')))
            flag=1  

        track_c = track[:grid_point_num]
        track_c=track_c.at[:, 0].set(480 - track_c[:, 0])#480 double of vc set 240
        depth_cv= img_depth(frame[:, ::-1])
        track_c_d=img_dep_cmb(first_track,depth_cv)

        R,angles=find_optimal_rotation(first_track_c_d, track_c_d)

        d_norm_now = np.linalg.norm(distance.squareform(distance.pdist(track_c, 'euclidean')))
        sf=d_norm_now/d_norm_first
        mtx1, mtx2, disparity = procrustes(first_track, track_c/sf)
        force = force + 0.5*(disparity-force)
        Data=[disparity*50000, mtx1-mtx2]
        print(Data)
    
        f_d_norm=0
        frame_bar=add_red_bar_to_frame(frame[:, ::-1], force*100)
       
        cv2.imshow("Point Tracking", frame_bar)
        cv2.imshow('Depth Estimation', depth_cv)


     

    key = cv2.waitKey(1)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("Point Tracking")
vc.release()