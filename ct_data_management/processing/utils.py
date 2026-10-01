import itertools
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from monai.visualize import blend_images
from time import time
from .pipeline import PipelinePart


class TimePipelinePart(PipelinePart):
    def __init__(self, pipeline_part):
        self._pipeline_part = pipeline_part

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        start_time = time()
        data, params = self._pipeline_part(data, params)
        end_time = time()
        print(type(self._pipeline_part).__name__, 'time:', str(end_time - start_time))
        return data, params


class InteractiveViewer(PipelinePart):
    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        ct_data = data.get('ct')

        # Collect all available segmentation types in display order.
        seg_tensors = [data[k] for k in ('nodule_seg', 'lung_seg', 'roi_seg')
                       if data.get(k) is not None]

        blended = ct_data
        if seg_tensors:
            colors = plt.get_cmap('tab10').colors
            color_cycle = itertools.cycle(colors)
            for seg_data in seg_tensors:
                for i in range(seg_data.size(0)):
                    blended = blend_images(
                        image=blended,
                        label=seg_data[i, ...][None, ...],
                        alpha=0.5,
                        cmap=ListedColormap(['none', next(color_cycle)])
                    )

        blended = blended.numpy().transpose(3, 2, 1, 0)[:, ::-1]
        z_max = blended.shape[0] - 1

        current_idx = 0

        fig, ax = plt.subplots()

        is_float = blended.max() <= 1.0
        im = ax.imshow(blended[current_idx], vmin=0, vmax=(1 if is_float else 255))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')

        title = ax.text(
            0.02, 0.98, f'Slice {current_idx}', transform=ax.transAxes,
            color='white', va='top', ha='left', fontsize=12, fontweight='bold'
        )

        def update_slice(step):
            nonlocal current_idx
            current_idx = np.clip(current_idx + step, 0, z_max)

            im.set_data(blended[current_idx])
            title.set_text(f'Slice {current_idx}')
            fig.canvas.draw_idle()

        def on_scroll(event):
            step = 1 if event.button == 'up' else -1
            update_slice(step)

        def on_key(event):
            if event.key in ['up', 'right']: update_slice(1)
            elif event.key in ['down', 'left']: update_slice(-1)

        fig.canvas.mpl_connect('scroll_event', on_scroll)
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()

        return data, params
