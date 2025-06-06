import os
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator, TubeWindowMaskingGenerator, RandomMaskingGenerator2D, RunningCellMaskingGenerator
from kinetics_av import VideoClsDataset, VideoMAE, VideoClsDatasetFrame
from ssv2 import SSVideoClsDataset


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        # me: new added
        if not args.no_augmentation:
            self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        else:
            print(f"==> Note: do not use 'GroupMultiScaleCrop' augmentation during pre-training!!!")
            self.train_augmentation = IdentityTransform()
        self.transform = transforms.Compose([
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type == 'part_window':
            print(f"==> Note: use 'part_window' masking generator (window_size={args.part_win_size[1:]}, apply_symmetry={args.part_apply_symmetry})")
            self.masked_position_generator = TubeWindowMaskingGenerator(
                args.window_size, args.mask_ratio, win_size=args.part_win_size[1:], apply_symmetry=args.part_apply_symmetry
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


class MaskGeneratorForAudio(object):
    def __init__(self, mask_type, input_size, mask_ratio):
        if mask_type == 'random':
            self.masked_position_generator = RandomMaskingGenerator2D(
                input_size, mask_ratio
            )
        else:
            raise NotImplementedError

    def __call__(self):
        return self.masked_position_generator()

    def __repr__(self):
        repr = "(MaskGeneratorForAudio,\n"
        # repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

class DataAugmentationForVideoMAEv2(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406] # IMAGENET_DEFAULT_MEAN
        self.input_std  = [0.229, 0.224, 0.225] # IMAGENET_DEFAULT_STD
        div             = True
        roll            = False
        normalize       = GroupNormalize(self.input_mean, self.input_std)

        # me: new added
        if not args.no_augmentation:
            self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66]) # We deploy
        else:
            print(f"==> Note: do not use 'GroupMultiScaleCrop' augmentation during pre-training!!!")
            self.train_augmentation = IdentityTransform()

        self.transform = transforms.Compose([
            self.train_augmentation,
            Stack(roll=roll),
            ToTorchFormatTensor(div=div),
            normalize,
        ])

        #----------------------------#
        # encoder masking
        #----------------------------#
        if args.mask_type == 'tube': # by default
            self.encoder_mask_map_generator = TubeMaskingGenerator(args.window_size, args.mask_ratio)

        elif args.mask_type == 'part_window': # no use
            print(f"==> Note: use 'part_window' masking generator (window_size={args.part_win_size[1:]}, apply_symmetry={args.part_apply_symmetry})")

            self.encoder_mask_map_generator = TubeWindowMaskingGenerator(
                args.window_size, args.mask_ratio, win_size=args.part_win_size[1:], apply_symmetry=args.part_apply_symmetry)
            
        else:
            raise NotImplementedError('Unsupported encoder masking strategy type.')
        
        #------------------------------#
        # decoder masking related
        # 面向视频, 采用run_cell方法
        #------------------------------#
        if args.decoder_mask_ratio > 0.:
            if args.decoder_mask_type == 'run_cell':
                self.decoder_mask_map_generator = RunningCellMaskingGenerator(args.window_size, args.decoder_mask_ratio)
                print('The decoder_mask_map_generator for video is deployed!!!') # NOTE: ok! using!
            else:
                raise NotImplementedError('Unsupported decoder masking strategy type.')

    def __call__(self, images):
        process_data, _  = self.transform(images)
        encoder_mask_map = self.encoder_mask_map_generator()

        #------------------------------------------------------#
        # 检查self是否存在decoder_mask_map_generator再进行操作
        #------------------------------------------------------#
        if hasattr(self, 'decoder_mask_map_generator'):
            decoder_mask_map = self.decoder_mask_map_generator() # we deploy
            # print('decoder_mask_map_generator is ok!!!') # NOTE: ok!!
        else:
            decoder_mask_map = 1 - encoder_mask_map

        return process_data, encoder_mask_map, decoder_mask_map # 返回三种类型的数据

    #------------------------------------------------------#
    # 在build_pretraining_dataset中被调用后, 会返回相关的信息
    #------------------------------------------------------#
    def __repr__(self):
        repr  = "(DataAugmentationForVideoMAEv2,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        
        repr += "  Encoder Masking Generator = %s,\n" % str(
            self.encoder_mask_map_generator)
        
        if hasattr(self, 'decoder_mask_map_generator'):
            repr += "  Decoder Masking Generator = %s,\n" % str(
                self.decoder_mask_map_generator)
        else:
            repr += "  Have not used decoder masking,\n"

        repr += ")"

        return repr


class MaskGeneratorForAudiov2(object):
    def __init__(self, mask_type, input_size, mask_ratio, decoder_mask_ratio, decoder_mask_type_audio):

        if mask_type == 'random':
            self.encoder_mask_map_generator = RandomMaskingGenerator2D(input_size, mask_ratio)
        else:
            raise NotImplementedError('Unsupported encoder masking strategy type for audio.')


        if decoder_mask_ratio > 0.:
            if decoder_mask_type_audio == 'random':
                self.decoder_mask_map_generator = RandomMaskingGenerator2D(input_size, decoder_mask_ratio)
                print('The decoder_mask_map_generator for audio is deployed!!!')
            else:
                raise NotImplementedError('Unsupported decoder masking strategy type for audio.')
            
    def __call__(self):
        return self.encoder_mask_map_generator(), self.decoder_mask_map_generator()

    def __repr__(self):
        repr  = "(MaskGeneratorForAudio,\n"
        repr += "encoder_mask_map_generator_for_audio = %s,\n" % str(self.encoder_mask_map_generator)
        repr += "decoder_mask_map_generator_for_audio = %s,\n" % str(self.decoder_mask_map_generator)
        repr += ")"
        return repr
    

def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAEv2(args) # for video

    mask_generator_audio = MaskGeneratorForAudiov2(
        mask_type=args.mask_type_audio,
        input_size=args.window_size_audio, # 计算出来的
        mask_ratio=args.mask_ratio_audio,
        decoder_mask_ratio=args.decoder_mask_ratio_audio, # new added
        decoder_mask_type_audio=args.decoder_mask_type_audio, # new added
    )

    dataset = VideoMAE(
        root=None,
        setting=args.data_path, # 也就是csv文件的地址
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform, # for video masking
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        # me: new added for VoxCeleb2
        # model=args.model,
        image_size=args.input_size, # 默认为160*160
        num_segments=args.num_samples, # 默认为1
        # me: for audio
        audio_conf=args.audio_conf,
        roll_mag_aug=args.roll_mag_aug,
        audio_sample_rate=args.audio_sample_rate,
        mask_generator_audio=mask_generator_audio, # for audio masking
    )

    print("Data Aug = %s" % str(transform)) # 打印输出视频转换的配置
    print("Data Aug for Audio = %s" % str(mask_generator_audio)) # 打印输出音频转换的配置

    return dataset


def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400
    
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51

    elif args.data_set == 'DFEW':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256, # me: actually no use
            new_width=320, # me: actually no use
            args=args)
        nb_classes = 7


    elif args.data_set == 'MAFW':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256, # me: actually no use
            new_width=320, # me: actually no use
            args=args,
            file_ext='png'
        )
        nb_classes = 11
        # for using 43 compound expressions
        if args.nb_classes == 43:
            nb_classes = args.nb_classes
            print(f"==> NOTE: using 43 compound expressions for MAFW!")


    elif args.data_set == 'RAVDESS':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256, # me: actually no use
            new_width=320, # me: actually no use
            args=args)
        nb_classes = 8


    elif args.data_set == 'CREMA-D':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256, # me: actually no use
            new_width=320, # me: actually no use
            args=args,
        )
        nb_classes = 6
        # for 4 basic emotions
        if args.nb_classes == 4:
            nb_classes = args.nb_classes
            print(f"==> NOTE: only using 4 emotions ('ANG', 'HAP', 'NEU', 'SAD')!")


    elif args.data_set == 'WEREWOLF-XL':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256, # me: actually no use
            new_width=320, # me: actually no use
            args=args,
            task='regression',
        )
        nb_classes = 3


    elif args.data_set == 'AVCAFFE':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256, # me: actually no use
            new_width=320, # me: actually no use
            args=args,
        )
        nb_classes = 5 # arousal or valence

    elif args.data_set == 'MSP-IMPROV':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,  # me: actually no use
            new_width=320,  # me: actually no use
            args=args,
        )
        nb_classes = 4
        print(f"==> NOTE: using 4 categorical emotions ('A', 'H', 'N', 'S')!")

    elif args.data_set == 'IEMOCAP':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,  # me: actually no use
            new_width=320,  # me: actually no use
            args=args,
        )
        nb_classes = 4


    elif args.data_set == 'MER2023':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256, # me: actually no use
            new_width=320, # me: actually no use
            args=args,
        )
        nb_classes = 6


    elif args.data_set == 'MER24-LABELED': # MER24-LABELED
        mode      = None
        anno_path = None

        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')

        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv')
            
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv')

        dataset = VideoClsDatasetFrame(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256, # me: actually no use
            new_width=320, # me: actually no use
            args=args,
        )
        nb_classes = 6


    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
