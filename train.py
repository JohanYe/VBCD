import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from accelerate import Accelerator
from models.crownmvm2 import CrownMVM,volume_to_point_cloud_tensor
from models.loss import curvature_penalty_loss
import os
import random
import numpy as np
import torch.nn.functional as F
import re
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_farthest_points
import pyvista as pv
# from mydataset.Dentaldataset import *
from dentaldataset import IOS_Datasetv2
from accelerate import DataLoaderConfiguration,DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
focal_loss = True
curvature_weight = 2
curvature_weighted_bce = False
dataloader_config = DataLoaderConfiguration(split_batches=True)
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(dataloader_config=dataloader_config,kwargs_handlers=[kwargs])

def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Training started.")

def create_run_directory(base_output_dir, exp_name=None):
    """
    Create a timestamped directory for the current run.
    Returns the path to the experiment directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if exp_name:
        run_dir = os.path.join(base_output_dir, f"{timestamp}_{exp_name}")
    else:
        run_dir = os.path.join(base_output_dir, timestamp)
    
    # Create main run directory
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'tensorboard'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'visualizations'), exist_ok=True)
    
    return run_dir

def save_visualization(outputs_pc, targets, pointcloud_inform, batch_y, step, vis_dir, rank, batch_idx=0):
    """Save point cloud and mesh visualizations for debugging"""
    # vis_dir is now the full path, just ensure it exists
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Only save from rank 0 to avoid conflicts
    if rank == 0:
        targets_np = targets.cpu().numpy()
        for i in range(len(outputs_pc)):
            try:
                mask = (batch_y == i)
                gtpoints = pointcloud_inform[mask][:,:3].cpu().numpy()
                
                # Save prediction point cloud
                point_cloud = pv.PolyData(outputs_pc[i].cpu().numpy() if torch.is_tensor(outputs_pc[i]) else outputs_pc[i])
                output_filename = os.path.join(vis_dir, f"pred_step{step}_batch{batch_idx}_sample{i}.ply")
                point_cloud.save(output_filename)
                
                # Save ground truth point cloud
                point_cloud_gt = pv.PolyData(gtpoints)
                gt_filename = os.path.join(vis_dir, f"gt_step{step}_batch{batch_idx}_sample{i}.ply")
                point_cloud_gt.save(gt_filename)
                
                logging.info(f"[Rank {rank}] Visualization saved: {output_filename}, {gt_filename}")
            except Exception as e:
                logging.warning(f"[Rank {rank}] Failed to save visualization for sample {i}: {str(e)}")


def cycle(dl):
    while True:
        for data in dl:
            yield data

def train(model, train_loader, val_loader, args, log_file, run_dir):
    model.to(accelerator.device)
    
    # Initialize TensorBoard writer only on main process
    writer = None
    if accelerator.is_main_process:
        tensorboard_dir = os.path.join(run_dir, 'tensorboard')
        writer = SummaryWriter(log_dir=tensorboard_dir)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_steps)
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    train_loader = cycle(train_loader)
    
    initial_step = 0
    step = initial_step
    best_val_dice = 0.0
    with tqdm(total=args.num_steps, desc="Training", unit="step", disable=not accelerator.is_main_process) as pbar:
        while step < args.num_steps + initial_step:
            model.train()
            total_loss = 0.0
            bce_loss_accum = 0.0
            normal_loss_accum = 0.0
            cpl_accum = 0.0
            
            for _ in range(args.accumulation_steps):
                inputs,targets,pointcloud_inform,batch_y,min_bound_crop,_ = next(train_loader)
                if curvature_weighted_bce:
                    curvatures = targets[:,-1,:,:,:]
                    non_zero_mask = curvatures != 0
                    curvatures_weighted = torch.where(non_zero_mask, 1+curvatures, curvatures)
                    criterition = nn.BCEWithLogitsLoss(weight=torch.exp(curvatures_weighted.unsqueeze(1)))
                else:
                    criterition = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0, device=accelerator.device))
                    
                with accelerator.autocast():
                    tooth_number = torch.tensor([3], device=inputs.device).long()
                    voxel_ind,refined_pos_with_normal,batch_x = model(inputs,tooth_number)
                    if step < 3 and accelerator.is_main_process:
                        torch.cuda.synchronize()
                        logging.info(f"[Rank {accelerator.process_index}] Forward pass completed")
                    if step % 2500 == 0:
                        logging.info(f'[Rank {accelerator.process_index}] The front points has {refined_pos_with_normal.shape[0]} points')
                    
                    bce_loss = criterition(voxel_ind,targets[:,:1,:,:,:])
                    cpl,normal_loss = curvature_penalty_loss(refined_pos_with_normal,pointcloud_inform,batch_x=batch_x,batch_y=batch_y) if step>100000 else (0,0)
                    refine_loss = 0.1*cpl + normal_loss
                    loss = bce_loss + refine_loss 
                    loss = loss/ args.accumulation_steps
                    
                    total_loss += loss.item()
                    bce_loss_accum += bce_loss.item() / args.accumulation_steps
                    normal_loss_accum += normal_loss.item() / args.accumulation_steps if isinstance(normal_loss, torch.Tensor) else normal_loss / args.accumulation_steps
                    cpl_accum += cpl.item() / args.accumulation_steps if isinstance(cpl, torch.Tensor) else cpl / args.accumulation_steps
                    
                accelerator.backward(loss)
           
            if accelerator.is_main_process:
                pbar.set_description(f'loss: {total_loss:.4f}')
            
            accelerator.wait_for_everyone()
            accelerator.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            accelerator.wait_for_everyone()    

            step += 1
            
            # TensorBoard logging on main process
            if accelerator.is_main_process and writer is not None:
                writer.add_scalar('loss/total', total_loss, step)
                writer.add_scalar('loss/bce', bce_loss_accum, step)
                writer.add_scalar('loss/normal', normal_loss_accum, step)
                writer.add_scalar('loss/chamfer', cpl_accum, step)
                writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], step)
      
            # Intermediate visualization every 1000 steps on main process
            if accelerator.is_main_process and step % 1000 == 0 and step > 0:
                model.eval()
                with torch.no_grad():
                    # Get a batch from training data for visualization
                    try:
                        inputs_vis, targets_vis, pc_info_vis, batch_y_vis, min_bound_vis, _ = next(train_loader)
                        tooth_number = torch.tensor([3], device=inputs_vis.device).long()
                        voxel_ind_vis, _, _ = model(inputs_vis, tooth_number)
                        position_indicator = F.sigmoid(voxel_ind_vis)
                        position_indicator = (position_indicator > 0.5).float()
                        
                        outputs_pc_vis = volume_to_point_cloud_tensor(
                            volume=position_indicator,
                            voxel_size=(0.15625,0.15625,0.15625),
                            origin=min_bound_vis.cpu()
                        )
                        
                        # Save visualizations
                        save_visualization(
                            outputs_pc_vis, 
                            targets_vis, 
                            pc_info_vis, 
                            batch_y_vis, 
                            step, 
                            os.path.join(run_dir, 'visualizations'),
                            accelerator.process_index
                        )
                        logging.info(f"[Rank {accelerator.process_index}] Visualization saved at step {step}")
                    except Exception as e:
                        logging.warning(f"[Rank {accelerator.process_index}] Visualization failed at step {step}: {str(e)}")
                model.train()
            
            if accelerator.is_main_process:
                if step % args.validation_interval == 0 or step == args.num_steps:
                    val_dice = validate(model, val_loader, step=step, save_path=os.path.join(run_dir, 'visualizations'), rank=accelerator.process_index)
                    logging.info(f"[Rank {accelerator.process_index}] Step [{step}/{args.num_steps}], Validation hausdorff: {val_dice:.4f}")
                    
                    # TensorBoard validation logging
                    if writer is not None:
                        writer.add_scalar('validation/hausdorff', val_dice, step)
                    
                    if step % args.validation_interval == 0:
                        valmodel = accelerator.unwrap_model(model)
                        step_model_path = os.path.join(run_dir, 'checkpoints', f"model_step_{step}.pth")
                        torch.save(valmodel.state_dict(), step_model_path)
                        logging.info(f"[Rank {accelerator.process_index}] Saved model at step {step}")
                    
                    if val_dice < best_val_dice:
                        best_val_dice = val_dice
                        valmodel = accelerator.unwrap_model(model)
                        best_model_path = os.path.join(run_dir, 'checkpoints', 'model_best.pth')
                        torch.save(valmodel.state_dict(), best_model_path)
                        logging.info(f"[Rank {accelerator.process_index}] Saved best model at step {step}")
                        
                if step % args.log_interval == 0:
                    logging.info(f"[Rank {accelerator.process_index}] Step {step} - BCE: {bce_loss_accum:.4f} - Normal: {normal_loss_accum:.4f} - Chamfer {cpl_accum:.4f}")
                    
            pbar.update(1)
    
    # Close TensorBoard writer
    if accelerator.is_main_process and writer is not None:
        writer.close()
        logging.info(f"[Rank {accelerator.process_index}] TensorBoard writer closed")


def dice_coefficient(tensor1, tensor2, epsilon=1e-6):
    assert tensor1.shape == tensor2.shape, "两个输入张量的形状必须相同"
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()

    intersection = (tensor1 * tensor2).sum(dim=(2, 3, 4))
    volumes_sum = tensor1.sum(dim=(2, 3, 4)) + tensor2.sum(dim=(2, 3, 4))

    dice = (2.0 * intersection + epsilon) / (volumes_sum + epsilon)
    return dice.mean()

def validate(model, val_loader, step, save_path='./visualizations', rank=0):
    model.eval()
    val_hausdorff = 0.0
    with torch.no_grad():
         
        for batch_idx,(inputs,targets,pointcloud_inform,batch_y,min_bound_crop,file_dir) in enumerate(val_loader):
           
            with accelerator.autocast():
                tooth_number = torch.tensor([3], device=inputs.device).long()
                voxel_ind,refined_pos_with_normal,batch_x = model(inputs,tooth_number)   
            position_indicator = F.sigmoid(voxel_ind)
            position_indicator = (position_indicator>0.5).float()

            outputs_pc = volume_to_point_cloud_tensor(volume=position_indicator,voxel_size=(0.15625,0.15625,0.15625),origin=min_bound_crop.cpu())
            hausdorff = dice_coefficient(position_indicator,targets[:,:1,:,:,:]).item()
            val_hausdorff += hausdorff
            if batch_idx < 2:
                targets_np = targets.cpu().numpy()
                for i in range(len(outputs_pc)):
                    mask = (batch_y == i)
                    gtpoints = pointcloud_inform[mask][:,:3].cpu().numpy()
                    point_cloud_gt = pv.PolyData(gtpoints)
                    point_cloud = pv.PolyData(outputs_pc[i].cpu().numpy() if torch.is_tensor(outputs_pc[i]) else outputs_pc[i])
                    output_filename = os.path.join(save_path, f"val_output_batch{batch_idx+1}_sample{i+1}_step{step}.ply")
                    gt_filename = os.path.join(save_path, f"val_gt_batch{batch_idx+1}_sample{i+1}_step{step}.ply")
                    point_cloud.save(output_filename)
                    point_cloud_gt.save(gt_filename)
                    logging.info(f"[Rank {rank}] Saved: {output_filename}")
                    logging.info(f"[Rank {rank}] Saved: {gt_filename}")
                
    val_hausdorff/=len(val_loader)
  
    return val_hausdorff

def load_data(batch_size=4, train_path='./train_data'):
    train_dataset = IOS_Datasetv2(train_path)
    val_dataset = IOS_Datasetv2(train_path,is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=8, shuffle=True,collate_fn=train_dataset.collate_fn)
    logging.info(f"Length of train dataloader {len(train_loader)}")
    val_loader = DataLoader(val_dataset, batch_size=2,num_workers=8, shuffle=False,collate_fn=train_dataset.collate_fn)
    logging.info(f"Length of val dataloader {len(val_loader)}")
    return train_loader, val_loader

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_steps', type=int, default=1000, help='Total number of steps for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Number of steps for gradient accumulation')
    parser.add_argument('--train_path', type=str, default='/train/PointCrown/PointCrown_v1.4.9', help='Path to training data')
    parser.add_argument('--val_path', type=str, default='/train/PointCrown/PointCrown_v1.4.9', help='Path to validation data')
    parser.add_argument('--validation_interval',type=int,default=4000,help='interval steps to validate')
    parser.add_argument('--continue_ckpt_dir',type=str,required=False,help='whether to use exist ckpt')
    parser.add_argument('--log_interval', type=int, default=50, help='Interval steps to log training information')
    parser.add_argument('--output_dir', type=str, default='./runs', help='Base directory for experiment runs')
    parser.add_argument('--exp_name', type=str, default=None, help='Optional experiment name for the run')
    args = parser.parse_args()
    
    # Create timestamped run directory
    run_dir = create_run_directory(args.output_dir, args.exp_name)
    
    # Log GPU rank information
    if accelerator.is_main_process:
        logging.info(f"Starting training on rank {accelerator.process_index} (total ranks: {accelerator.num_processes})")
        logging.info(f"Run directory: {run_dir}")
    
    model = CrownMVM(in_channels=1,out_channels=4)
    if args.continue_ckpt_dir:
        ckpt = torch.load(args.continue_ckpt_dir)
        ckpt = {k[7:] if k.startswith('module.') else k: v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        if accelerator.is_main_process:
            logging.info('Load checkpoint complete')
    
    log_file = os.path.join(run_dir, 'training.log')
    setup_logging(log_file)
    if accelerator.is_main_process:
        logging.info(f"GPU Rank: {accelerator.process_index}/{accelerator.num_processes}")
        logging.info(f"Run output saved to: {run_dir}")
    
    train_loader, val_loader = load_data(batch_size=args.batch_size, train_path=args.train_path)
    train(model, train_loader, val_loader, args, log_file=log_file, run_dir=run_dir)
if __name__== "__main__":
    seed_everything(42)
    main()