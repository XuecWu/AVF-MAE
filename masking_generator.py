import numpy as np


class RandomMaskingGenerator2D:
    def __init__(self, input_size, mask_ratio):
        self.height, self.width = input_size
        self.num_patches =  self.height * self.width
        self.num_masks = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}".format(
            self.num_patches, self.num_masks
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_masks), # 0: for unmasked
            np.ones(self.num_masks), # 1: for masked
        ])
        np.random.shuffle(mask) # in-place
        return mask


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask


class TubeWindowMaskingGenerator:
    def __init__(self, input_size, mask_ratio, win_size, apply_symmetry=None):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

        assert self.width == win_size[1] and win_size[1] % 2 == 0, "Error: window width must be equal to input width and be even if apply windown attn and face symmetrical masking."

        assert self.height % win_size[0] == 0 and self.width % win_size[1] == 0
        self.spatial_part_size = (self.height // win_size[0], self.width // win_size[1])
        self.num_wins_per_frame = self.spatial_part_size[0] * self.spatial_part_size[1]
        self.num_patches_per_win = win_size[0] * win_size[1]
        self.num_masks_per_win = int(mask_ratio * self.num_patches_per_win)
        self.num_unmasks_per_win = self.num_patches_per_win - self.num_masks_per_win
        self.apply_symmetry = apply_symmetry
        if apply_symmetry is not None:
            assert apply_symmetry in ['global', 'local']
            assert mask_ratio > 0.5
            self.win_width_half = (win_size[1] // 2)
            self.num_patches_per_win_half = win_size[0] * self.win_width_half
            self.win_size_half = (win_size[0], self.win_width_half)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = []
        if self.apply_symmetry is None:
            for i in range(self.num_wins_per_frame):
                mask_per_win = [0] * (self.num_patches_per_win - self.num_masks_per_win) + [1] * self.num_masks_per_win
                np.random.shuffle(mask_per_win)
                mask_per_frame.extend(mask_per_win)
        elif self.apply_symmetry == 'global':
            left = np.random.rand() < 0.5
            mask_per_frame = []
            for i in range(self.num_wins_per_frame):
                mask_per_win_half = [0] * self.num_unmasks_per_win + [1] * (self.num_patches_per_win_half - self.num_unmasks_per_win)
                np.random.shuffle(mask_per_win_half)
                mask_per_win = []
                for i in range(self.win_size_half[0]):
                    if left:
                        mask_per_win.extend([mask_per_win_half[i*self.win_size_half[1]:(i+1)*self.win_size_half[1]] + [1] * self.win_size_half[1]])
                    else:
                        mask_per_win.extend([[1] * self.win_size_half[1] + mask_per_win_half[i*self.win_size_half[1]:(i+1)*self.win_size_half[1]]])
                # if left:
                #     # mask_per_win = np.hstack([np.array(mask_per_win_half).reshape(self.win_size_half), np.ones(self.win_size_half)])
                #     mask_per_win = [mask_per_win_half[i*self.win_size_half[1]:(i+1)*self.win_size_half[1]] + [1] * self.win_size_half[1] for i in self.win_size_half[0]]
                # else:
                #     # mask_per_win = np.hstack([np.ones(self.win_size_half), np.array(mask_per_win_half).reshape(self.win_size_half)])
                #     mask_per_win = [[1] * self.win_size_half[1] + mask_per_win_half[i*self.win_size_half[1]:(i+1)*self.win_size_half[1]] for i in self.win_size_half[0]]
                # mask_per_frame.append(mask_per_win.flatten())
                mask_per_frame.extend(mask_per_win)
            # mask_per_frame = np.hstack(mask_per_frame)
        else: # local
            mask_per_frame = []
            for i in range(self.num_wins_per_frame):
                mask_per_win_half = [0] * self.num_unmasks_per_win + [1] * (self.num_patches_per_win_half - self.num_unmasks_per_win)
                np.random.shuffle(mask_per_win_half)
                left = np.random.rand() < 0.5
                mask_per_win = []
                for i in range(self.win_size_half[0]):
                    if left:
                        mask_per_win.extend([mask_per_win_half[i*self.win_size_half[1]:(i+1)*self.win_size_half[1]] + [1] * self.win_size_half[1]])
                    else:
                        mask_per_win.extend([[1] * self.win_size_half[1] + mask_per_win_half[i*self.win_size_half[1]:(i+1)*self.win_size_half[1]]])
                mask_per_frame.extend(mask_per_win)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask


class Cell():
    def __init__(self, num_masks, num_patches):
        self.num_masks   = num_masks
        self.num_patches = num_patches
        self.size        = num_masks + num_patches
        self.queue       = np.hstack([np.ones(num_masks), np.zeros(num_patches)])
        self.queue_ptr   = 0

    def set_ptr(self, pos=-1):
        self.queue_ptr = np.random.randint(self.size) if pos < 0 else pos

    def get_cell(self):
        cell_idx = (np.arange(self.size) + self.queue_ptr) % self.size
        return self.queue[cell_idx]

    def run_cell(self):
        self.queue_ptr += 1


class RunningCellMaskingGenerator:
    def __init__(self, input_size, mask_ratio=0.5):
        self.frames, self.height, self.width = input_size
        self.mask_ratio                      = mask_ratio

        num_masks_per_cell   = int(4 * self.mask_ratio)
        assert 0 < num_masks_per_cell < 4
        num_patches_per_cell = 4 - num_masks_per_cell

        self.cell      = Cell(num_masks_per_cell, num_patches_per_cell)
        self.cell_size = self.cell.size

        mask_list = []
        for ptr_pos in range(self.cell_size):
            self.cell.set_ptr(ptr_pos)
            mask = []
            for _ in range(self.frames):
                self.cell.run_cell()
                mask_unit = self.cell.get_cell().reshape(2, 2)
                mask_map  = np.tile(mask_unit,
                                   [self.height // 2, self.width // 2])
                mask.append(mask_map.flatten())
            mask = np.stack(mask, axis=0)
            mask_list.append(mask)
        self.all_mask_maps = np.stack(mask_list, axis=0)

    def __repr__(self):
        repr_str = f"Running Cell Masking with mask ratio {self.mask_ratio}"
        return repr_str

    def __call__(self):
        mask = self.all_mask_maps[np.random.randint(self.cell_size)]
        return np.copy(mask)