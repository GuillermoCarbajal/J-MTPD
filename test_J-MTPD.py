import torch
from skimage.io import imread
from skimage.transform import rescale
from utils_bd.homographies import compute_intrinsics, generarK, mostrar_kernels, generate_video, get_offsets_from_positions, \
                                     show_positions_found, sort_positions, save_kernels_from_offsets, show_kernels_from_offsets_on_blurry_image
import numpy as np
from utils_bd.RL_restoration_from_positions import RL_restore_from_positions, combined_RL_restore_from_positions
from models.network_nimbusr_pmbm import NIMBUSR_PMBM as net
from models.network_nimbusr_offsets import NIMBUSR_Offsets as net_nimbusr_offsets

from utils_bd.visualization import save_image, tensor2im, save_kernels_grid, save_video
import os 
import argparse
from models.CameraShakeModelTwoBranches import CameraShakeModelTwoBranches as TwoBranches
from models.CameraShakeModelThreeBranches import CameraShakeModelThreeBranches as ThreeBranches
import json
from PIL import Image
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--blurry_image', '-b', type=str, help='blurry image', default='/home/guillermo/github/camera_shake/data/COCO_homographies_small_gf1/blurry/000000000009_0.jpg')
parser.add_argument('--reblur_model', '-m', type=str, help='reblur model', required=True)
parser.add_argument('--restoration_network', '-rn', type=str, help='restoration network', default=r'NIMBUSR/model_zoo/PMPB_220000_G.pth')
parser.add_argument('--rescale_factor','-rf', type=float, default=1)
parser.add_argument('--restoration_method','-rm', type=str, default='NIMBUSR')
parser.add_argument('--nimbusr_model_type','-nmt', type=str, default='offsets')
parser.add_argument('--output_folder','-o', type=str, default='results_blind_pmpb')
parser.add_argument('--architecture','-a', type=str, default='two_branches')
parser.add_argument('--crop', action='store_true', default=False, help='whether to compute the kernels from central crop')
parser.add_argument('--crop_size','-cs', type=int, default=160)
parser.add_argument('--scale_pitch_yaw', action='store_true', help='whether to scale the pitch and yaw', default=False)
parser.add_argument('--scale_roll', action='store_true', help='whether to scale the roll', default=False)
parser.add_argument('--superresolution_factor','-sf', type=int, default=1)
parser.add_argument('--save_video', action='store_true', help='whether to save the video or not', default=False)
parser.add_argument('--bordersize', type=int, default=32)
parser.add_argument('--convolution', action='store_true', help='whether to flip the offsets before inputing to netG', default=False)
parser.add_argument('--focal_length', '-f', type=float, help='given focal length', default=0)
parser.add_argument('--given_focal_length', action='store_true', help='whether to use the focal length in the forward pass', default=False)
parser.add_argument('--offsets_BT', action='store_true', help='whether to pass offsets computed from the adjunct operator to the restorer', default=False)

args = parser.parse_args()

GPU = 0

def load_nimbusr_net(type='offsets'):
    opt_net = { "n_iter": 8
        , "h_nc": 64
        , "in_nc": 4 #2 if args.gray else 4 #4
        , "out_nc":3 #1 if args.gray else 3 #3
        #, "ksize": 25
        , "nc": [64, 128, 256, 512]
        , "nb": 2
        , "gc": 32
        , "ng": 2
        , "reduction" : 16
        , "act_mode": "R" 
        , "upsample_mode": "convtranspose" 
        , "downsample_mode": "strideconv"}

    path_pretrained = args.restoration_network #r'../model_zoo/NIMBUSR.pth'
    
    if type=='pmbm':
        netG = net(n_iter=opt_net['n_iter'],
                    h_nc=opt_net['h_nc'],
                    in_nc=opt_net['in_nc'],
                    out_nc=opt_net['out_nc'],
                    nc=opt_net['nc'],
                    nb=opt_net['nb'],
                    act_mode=opt_net['act_mode'],
                    downsample_mode=opt_net['downsample_mode'],
                    upsample_mode=opt_net['upsample_mode']
                    )
    elif type=='offsets':
        netG = net_nimbusr_offsets(n_iter=opt_net['n_iter'],
            h_nc=opt_net['h_nc'],
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            nc=opt_net['nc'],
            nb=opt_net['nb'],
            act_mode=opt_net['act_mode'],
            downsample_mode=opt_net['downsample_mode'],
            upsample_mode=opt_net['upsample_mode']
            )

    if os.path.exists(path_pretrained):
        print('Loading model for G [{:s}] ...'.format(path_pretrained))
        netG.load_state_dict(torch.load(path_pretrained))
    else:
        print('Model does not exists')
        
    netG = netG.to('cuda')

    return netG
    
if args.blurry_image.endswith('.txt'):
    with open(args.blurry_image) as f:
        blurry_images_list =  f.readlines()
        blurry_images_list = [file[:-1] for file in blurry_images_list]
        #blurry_images_list = blurry_images_list[48:]
else:
    blurry_images_list = [args.blurry_image]
    

if args.restoration_method=='NIMBUSR':
    netG = load_nimbusr_net(args.nimbusr_model_type)
    netG.eval()
    noise_level = 0.01
    noise_level = torch.FloatTensor([noise_level]).view(1,1,1).cuda(GPU)  
elif args.restoration_method=='RL':
    n_iters = 20   

reblur_model = args.reblur_model 
#sharp_image_filename = '/home/guillermo/github/camera_shake/data/COCO_homographies_small_gf1//sharp/000000000009_0.jpg'

n_positions = 25
restoration_method=args.restoration_method
output_folder=args.output_folder

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
with open(os.path.join(args.output_folder, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)


if args.architecture == 'two_branches':
    #intrinsics = compute_intrinsics(W*args.superresolution_factor, H*args.superresolution_factor, args.bordersize).cuda(GPU)[None]
    camera_model = TwoBranches().cuda(GPU)
elif args.architecture == 'three_branches':
    camera_model = ThreeBranches().cuda(GPU)

state_dict = torch.load(reblur_model)
camera_model.load_state_dict(state_dict, strict=False)
camera_model.eval()

for blurry_image_filename in blurry_images_list:    


    print(blurry_image_filename)
    blurry_image = rescale(imread(blurry_image_filename)/255.0, (args.rescale_factor,args.rescale_factor,1),anti_aliasing=True)
    #sharp_image = rescale(imread(sharp_image_filename)/255.0,(0.6,0.6,1),anti_aliasing=True)
    blurry_tensor = torch.from_numpy(blurry_image).permute(2,0,1)[None].cuda(GPU).float()
    #sharp_tensor = torch.from_numpy(sharp_image).permute(2,0,1)[None].cuda(GPU).float()
    initial_tensor = blurry_tensor.clone()
    
    _, C,H,W = blurry_tensor.shape
    print(C,H,W)

    
    with torch.no_grad():
        if args.crop:
            camera_positions = camera_model(blurry_tensor[:,:,H//2-args.crop_size//2:H//2+args.crop_size//2, 
                                                          W//2-args.crop_size//2:W//2+args.crop_size//2] - 0.5)
        else:
            if args.architecture == 'two_branches':

                if args.focal_length > 0:
                    f = torch.Tensor([args.focal_length]).to(blurry_tensor.device)
                    #f = torch.Tensor([float(max(H,W))]).to(tensor_img.device) 
                    intrinsics = torch.Tensor([[f, 0, W/2],[0, f, H/2], [0, 0, 1] ]).cuda(blurry_tensor.device)
                    intrinsics = intrinsics[None,:,:]
                else:
                    intrinsics = compute_intrinsics(W,H).cuda(GPU)[None]
                    f =  torch.Tensor([max(H,W)]).to(blurry_tensor.device)
                    
                    #focal_channel = f/maximo * torch.ones(N,1,H,W).to(tensor_img.device)
                    #cam_input = torch.concat((focal_channel, tensor_img), dim=1)

                if args.given_focal_length:
                     camera_positions = camera_model(blurry_tensor - 0.5,f)
                else:
                    camera_positions = camera_model(blurry_tensor - 0.5)

            elif args.architecture == 'three_branches':
                camera_positions, intrinsics =  camera_model(blurry_tensor - 0.5)
                #intrinsics[0,0,0]=W
                #intrinsics[0,1,1]=H


            #camera_positions[:,:,0]=camera_positions[:,:,0]*(W/360)
            #camera_positions[:,:,1]=camera_positions[:,:,1]*(H/360)
        if args.scale_pitch_yaw:
            camera_positions[:,:,0] = camera_positions[:,:,0]/W
            camera_positions[:,:,1] = camera_positions[:,:,1]/H
        if args.scale_roll:
            camera_positions[:,:,2]=camera_positions[:,:,2]/np.sqrt(H*H+W*W)   

    #camera_positions = torch.from_numpy(camera_positions_np).cuda(GPU)[None].float()
    
    order = sort_positions(camera_positions[0])
    camera_positions[0] = camera_positions[0,order,:]



    if restoration_method=='RL':
        output = RL_restore_from_positions(blurry_tensor, initial_tensor, camera_positions, 
                                        n_iters, GPU, isDebug=True, reg_factor=0)
        #combined_RL_restore_from_positions(blurry_tensor, initial_tensor, camera_positions, n_iters, GPU, isDebug=True)
    else: 
        with torch.no_grad():
            if args.nimbusr_model_type=='pmbm':
                output = netG(blurry_tensor, camera_positions, intrinsics, args.superresolution_factor, sigma=noise_level[None,:])
            elif args.nimbusr_model_type=='offsets':
                offsets = get_offsets_from_positions(blurry_tensor.shape, camera_positions, intrinsics)
                offsets = offsets.reshape(1,2*n_positions, H,W)
                offsets_BT=None
                if args.offsets_BT:
                    offsets_BT = get_offsets_from_positions(blurry_tensor.shape, camera_positions, intrinsics, adjunct_operator_offsets=True)
                    offsets_BT = offsets_BT.reshape(1,2*n_positions, H,W)

                #if args.convolution:
                #    netG_input = -offsets   
                #else:
                #    netG_input = offsets 

                output = netG(blurry_tensor, offsets, 1, sigma=noise_level[None,:], offsets_BT=offsets_BT)
        

    img_name, ext = blurry_image_filename.split('/')[-1].split('.')    
    
    output_img = tensor2im(torch.clamp(output[0].detach(),0,1) - 0.5)
    save_image(output_img, os.path.join(output_folder, img_name + '_PMBM.png' ))
    
    found_positions_np = camera_positions[0].detach().cpu().numpy()
    np.savetxt(os.path.join(output_folder,f'{img_name}_found_positions.txt'), found_positions_np)
    pose = np.zeros((found_positions_np.shape[0], 6))
    pose[:, 3:] = found_positions_np
    
    K, _ = generarK((H,W,C), pose, A=intrinsics[0].detach().cpu().numpy())
    kernels_estimated = mostrar_kernels(K, (H,W,C), output_name = os.path.join(output_folder, img_name + '_kernels_found.png' ))

    show_kernels_from_offsets_on_blurry_image(blurry_tensor[0],offsets[0].reshape(n_positions,2,H,W), os.path.join(output_folder, img_name + '_kernels.png' ))
    
    save_image((255*blurry_image).astype(np.uint8), os.path.join(output_folder, img_name + '.png' ))
    print('Output saved in ', os.path.join(output_folder, img_name + '_' + restoration_method + '.png' ))
    
    frames, reblurred = generate_video(output, camera_positions, intrinsics[0])
    print(reblurred.shape)
    reblurred = tensor2im(torch.clamp(reblurred[0].detach(),0,1) - 0.5)
    save_image(reblurred, os.path.join(output_folder, img_name + '_reblurred.png' ))

    show_positions_found(found_positions_np, intrinsics[0,0,0].detach().cpu().numpy(), os.path.join(output_folder, img_name + '_positions_found.png'))
    if args.offsets_BT:
        save_kernels_from_offsets(offsets_BT,os.path.join(output_folder, img_name + '_kernels_BT.png' ))
        save_kernels_from_offsets(-offsets,os.path.join(output_folder, img_name + '_kernels_neg.png' ))

 
    
    if args.save_video:
        #imgs=[]; 
        output_video = os.path.join(output_folder, img_name + '.avi')
        save_video(frames, output_video)
        
        # cv_images=[]
        # position = (W-100, 50)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 1.5
        # color = (0, 255, 0)
        # thickness = 2
        # #blurry_to_draw = cv2.cvtColor((255*blurry_image).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # #blurry_to_draw = cv2.putText(cv2.UMat(blurry_to_draw), 'Blurry', position, font, font_scale, color, thickness)
        # #cv_images.append(blurry_to_draw)
        # for n in range(len(frames)):
        #     sharp_n = tensor2im(torch.clamp(frames[n][0].detach(),0,1) - 0.5)
        #     sharp_n_draw = cv2.putText(cv2.UMat(sharp_n), str(n), position, font, font_scale, color, thickness)
        #     #imgs.append(Image.fromarray(np.uint8(sharp_n)).convert("P",palette=Image.ADAPTIVE))
        #     cv_images.append(cv2.cvtColor(sharp_n_draw, cv2.COLOR_RGB2BGR))
        #     #save_image(sharp_n, os.path.join(output_folder, img_name + f'_{n}.png' ))
        
        # #imgs[0].save(fp=os.path.join(output_folder, img_name + '.gif'), format='GIF', append_images=imgs,
        # #         save_all=True, duration=30, loop=0)
    
        # video = cv2.VideoWriter(os.path.join(output_folder, img_name + '.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 25, (W,H))
        # for image in cv_images:
        #     video.write(image)
        