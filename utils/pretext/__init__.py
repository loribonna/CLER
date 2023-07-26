import argparse
from utils.pretext.jigsaw import JigsawPretext
from utils.pretext.rotation import RotationPretext
from utils.pretext.jig_rot_pretext import JigRotPretext
from utils.pretext.vflip import VFlipPretext
from utils.pretext.rel_patch_loc import RelPatchLocPretext

def get_pretext_args(parser):
    tmp_parser = argparse.ArgumentParser()    
    tmp_parser.add_argument('--pretext', type=str, default='jigsaw', choices=['jigsaw', 'rotation', 'jig_rot', 'vflip', 'rel_patch_loc'],
                        help='jigsaw|rotation|jig_rot|vflip|rel_patch_loc')

    parser.add_argument('--pretext', type=str, default='jigsaw', choices=['jigsaw', 'rotation','jig_rot', 'vflip', 'rel_patch_loc'],
                        help='jigsaw|rotation|jig_rot|vflip|rel_patch_loc')
    args, _ = tmp_parser.parse_known_args()

    if args.pretext is None:
        return parser
    elif args.pretext == 'rotation':
        return RotationPretext.update_parser(parser)
    elif args.pretext == 'jigsaw':
        return JigsawPretext.update_parser(parser)
    elif args.pretext == 'vflip':
        return VFlipPretext.update_parser(parser)
    elif args.pretext == 'rel_patch_loc':
        return RelPatchLocPretext.update_parser(parser)
    elif args.pretext == 'jig_rot':
        return JigRotPretext.update_parser(parser)
    else:
        raise ValueError('Unknown pretext task: {}'.format(args.pretext))


def get_pretext_task(args):
    if not hasattr(args, 'pretext'):
        return None
    elif args.pretext == 'rotation':
        return RotationPretext(args)
    elif args.pretext == 'jigsaw':
        return JigsawPretext(args)
    elif args.pretext == 'vflip':
        return VFlipPretext(args)
    elif args.pretext == 'rel_patch_loc':
        return RelPatchLocPretext(args)
    elif args.pretext == 'jig_rot':
        return JigRotPretext(args)
    else:
        raise ValueError('Unknown pretext task: {}'.format(args.pretext))