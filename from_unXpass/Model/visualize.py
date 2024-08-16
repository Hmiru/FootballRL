import torch
from torch.utils.data import DataLoader
import pandas as pd
from matplotlib import pyplot as plt
from mplsoccer import Pitch
from from_unXpass.Model.Soccermap import PytorchSoccerMapModel
from from_unXpass.dataset.dataloader import SoccerDataset
from from_unXpass.dataset.Into_Soccermap_tensor import ToSoccerMapTensor
from from_unXpass.dataset.Into_Soccermap_tensor import *
"""Data visualisation."""


def plot_action(
        action: pd.Series,
        surface=None,
        show_action=True,
        show_visible_area=False,
        ax=None,
        surface_kwargs={},
) -> None:
    """Plot a SPADL action with 360 freeze frame.

    Parameters
    ----------
    action : pandas.Series
        A row from the actions DataFrame.
    surface : np.arry, optional
        A surface to visualize on top of the pitch.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on.
    surface_kwargs : dict, optional
        Keyword arguments to pass to the surface plotting function.
    """
    # parse freeze frame
    freeze_frame = pd.DataFrame.from_records(action["freeze_frame_360"])
    teammate_locs = freeze_frame[freeze_frame.teammate]
    opponent_locs = freeze_frame[~freeze_frame.teammate]

    # set up pitch
    p = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68)
    if ax is None:
        _, ax = p.draw(figsize=(12, 8))
    else:
        p.draw(ax=ax)

    # plot action
    if show_action:
        p.arrows(
            action["start_x"],
            action["start_y"],
            action["end_x"],
            action["end_y"],
            color="black",
            headwidth=5,
            headlength=5,
            width=1,
            ax=ax,
        )

    # plot freeze frame
    p.scatter(teammate_locs.x, teammate_locs.y, c="#6CABDD", s=80, ec="k", ax=ax)
    p.scatter(opponent_locs.x, opponent_locs.y, c="#C8102E", s=80, ec="k", ax=ax)
    p.scatter(action["start_x"], action["start_y"], c="w", s=40, ec="k", ax=ax)

    # plot surface
    if surface is not None:
        ax.imshow(surface, extent=[0.0, 105.0, 0.0, 68.0], origin="lower", **surface_kwargs)

    return ax


if __name__ == "__main__":
    model_path=\
        "/home/crcteam/PycharmProjects/Football_RL/from_unXpass/Model/lightning_logs/version_54/checkpoints/epoch=0-step=643.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    final_df = pd.read_csv("../dataset/total_data_with_state_label_mask.csv", index_col=0)
    row_data = final_df.iloc[3]
    sample = convert_row_to_sample(row_data)
    # Convert to tensor format
    tensor_converter = ToSoccerMapTensor()
    matrix, mask, target = tensor_converter(sample)

    matrix = matrix.unsqueeze(0).to(device)
    #
    # Load the model from checkpoint and move it to the GPU (if available)
    model = PytorchSoccerMapModel.load_from_checkpoint(model_path)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        surface = model(matrix)
        print(surface.shape)
        surface = surface.squeeze(0).squeeze(0).cpu().numpy()  # Move the result back to CPU for visualization
        #결과 텐서에서 불필요한 차원을 제거하고, 시각화를 위해 CPU로 이동하여 NumPy 배열로 변환합니다.

    original_action = pd.Series(sample)
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_action(original_action, surface, ax=ax)
    plt.show()
