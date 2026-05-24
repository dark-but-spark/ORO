import gc  # Added for aggressive garbage collection
from csv import writer

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in semantic segmentation
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Arguments:
        alpha {float} -- Weighting factor for positive/negative samples (default: 0.25)
        gamma {float} -- Focusing parameter to down-weight easy examples (default: 2.0)
        reduction {str} -- Reduction type: 'mean', 'sum', or 'none' (default: 'mean')
    
    Reference:
        Lin, T.Y., Goyal, P., Girshick, R., He, K. and Dollar, P., 2017. 
        Focal loss for dense object detection. ICCV 2017.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Sigmoid activation
        p = torch.sigmoid(inputs)
        
        # Compute binary cross entropy with logits (more numerically stable)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute focal weight: (1 - p_t)^gamma
        # p_t = p * target + (1 - p) * (1 - target)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Final focal loss
        focal_loss = alpha_factor * focal_weight * bce_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for direct optimization of Dice coefficient
    Works well for imbalanced segmentation tasks
    
    Arguments:
        smooth {float} -- Smoothing factor to avoid division by zero (default: 1.0)
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Sigmoid activation
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Compute Dice coefficient
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined Loss (BCE/Focal + Dice) for balanced training
    
    Arguments:
        bce_weight {float} -- Weight for BCE/Focal loss (default: 0.5)
        dice_weight {float} -- Weight for Dice loss (default: 0.5)
        use_focal {bool} -- Use Focal Loss instead of BCE (default: False)
        alpha {float} -- Focal Loss alpha parameter (default: 0.25)
        gamma {float} -- Focal Loss gamma parameter (default: 2.0)
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, use_focal=False, alpha=0.25, gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        if use_focal:
            self.bce_loss = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
        
        self.dice_loss = DiceLoss(smooth=1.0)
    
    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.bce_weight * bce + self.dice_weight * dice


class Conv2d_batchnorm(torch.nn.Module):
	'''
	2D Convolutional layers

	Arguments:
		num_in_filters {int} -- number of input filters
		num_out_filters {int} -- number of output filters
		kernel_size {tuple} -- size of the convolving kernel
		stride {tuple} -- stride of the convolution (default: {(1, 1)})
		activation {str} -- activation function (default: {'relu'})

	'''
	def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = (1,1), activation = 'relu'):
		super().__init__()
		self.activation = activation
		self.conv1 = torch.nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size, stride=stride, padding = 'same')
		self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
	
	def forward(self,x):
		x = self.conv1(x)
		x = self.batchnorm(x)
		
		if self.activation == 'relu':
			return torch.nn.functional.relu(x)
		else:
			return x


class Multiresblock(torch.nn.Module):
	'''
	MultiRes Block
	
	Arguments:
		num_in_channels {int} -- Number of channels coming into mutlires block
		num_filters {int} -- Number of filters in a corrsponding UNet stage
		alpha {float} -- alpha hyperparameter (default: 1.67)
	
	'''

	def __init__(self, num_in_channels, num_filters, alpha=1.67):
	
		super().__init__()
		self.alpha = alpha
		self.W = num_filters * alpha
		
		filt_cnt_3x3 = int(self.W*0.167)
		filt_cnt_5x5 = int(self.W*0.333)
		filt_cnt_7x7 = int(self.W*0.5)
		num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7
		
		self.shortcut = Conv2d_batchnorm(num_in_channels ,num_out_filters , kernel_size = (1,1), activation='None')

		self.conv_3x3 = Conv2d_batchnorm(num_in_channels, filt_cnt_3x3, kernel_size = (3,3), activation='relu')

		self.conv_5x5 = Conv2d_batchnorm(filt_cnt_3x3, filt_cnt_5x5, kernel_size = (3,3), activation='relu')
		
		self.conv_7x7 = Conv2d_batchnorm(filt_cnt_5x5, filt_cnt_7x7, kernel_size = (3,3), activation='relu')

		self.batch_norm1 = torch.nn.BatchNorm2d(num_out_filters)
		self.batch_norm2 = torch.nn.BatchNorm2d(num_out_filters)

	def forward(self,x):

		shrtct = self.shortcut(x)
		
		a = self.conv_3x3(x)
		b = self.conv_5x5(a)
		c = self.conv_7x7(b)

		x = torch.cat([a,b,c],axis=1)
		x = self.batch_norm1(x)

		x = x + shrtct
		x = self.batch_norm2(x)
		x = torch.nn.functional.relu(x)
	
		return x


class Respath(torch.nn.Module):
	'''
	ResPath
	
	Arguments:
		num_in_filters {int} -- Number of filters going in the respath
		num_out_filters {int} -- Number of filters going out the respath
		respath_length {int} -- length of ResPath
		
	'''

	def __init__(self, num_in_filters, num_out_filters, respath_length):
	
		super().__init__()

		self.respath_length = respath_length
		self.shortcuts = torch.nn.ModuleList([])
		self.convs = torch.nn.ModuleList([])
		self.bns = torch.nn.ModuleList([])

		for i in range(self.respath_length):
			if(i==0):
				self.shortcuts.append(Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (1,1), activation='None'))
				self.convs.append(Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (3,3),activation='relu'))

				
			else:
				self.shortcuts.append(Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation='None'))
				self.convs.append(Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation='relu'))

			self.bns.append(torch.nn.BatchNorm2d(num_out_filters))
		
	
	def forward(self,x):

		for i in range(self.respath_length):

			shortcut = self.shortcuts[i](x)

			x = self.convs[i](x)
			x = self.bns[i](x)
			x = torch.nn.functional.relu(x)

			x = x + shortcut
			x = self.bns[i](x)
			x = torch.nn.functional.relu(x)

		return x


class MultiResUnet(torch.nn.Module):
	'''
	MultiResUNet
	
	Arguments:
		input_channels {int} -- number of channels in image
		num_classes {int} -- number of segmentation classes
		alpha {float} -- alpha hyperparameter (default: 1.67)
		dropout_rate {float} -- dropout rate for regularization (default: 0.2)
	
	Returns:
		[keras model] -- MultiResUNet model
	'''
	def __init__(self, input_channels, num_classes, alpha=1.67, dropout_rate=0.2):
		super().__init__()
		
		self.alpha = alpha
		self.dropout_rate = dropout_rate
		
		# Encoder Path
		self.multiresblock1 = Multiresblock(input_channels,32)
		self.in_filters1 = int(32*self.alpha*0.167)+int(32*self.alpha*0.333)+int(32*self.alpha* 0.5)
		self.pool1 =  torch.nn.MaxPool2d(2)
		self.respath1 = Respath(self.in_filters1,32,respath_length=4)

		self.multiresblock2 = Multiresblock(self.in_filters1,32*2)
		self.in_filters2 = int(32*2*self.alpha*0.167)+int(32*2*self.alpha*0.333)+int(32*2*self.alpha* 0.5)
		self.pool2 =  torch.nn.MaxPool2d(2)
		self.respath2 = Respath(self.in_filters2,32*2,respath_length=3)
	
	
		self.multiresblock3 =  Multiresblock(self.in_filters2,32*4)
		self.in_filters3 = int(32*4*self.alpha*0.167)+int(32*4*self.alpha*0.333)+int(32*4*self.alpha* 0.5)
		self.pool3 =  torch.nn.MaxPool2d(2)
		self.respath3 = Respath(self.in_filters3,32*4,respath_length=2)
	
	
		self.multiresblock4 = Multiresblock(self.in_filters3,32*8)
		self.in_filters4 = int(32*8*self.alpha*0.167)+int(32*8*self.alpha*0.333)+int(32*8*self.alpha* 0.5)
		self.pool4 =  torch.nn.MaxPool2d(2)
		self.respath4 = Respath(self.in_filters4,32*8,respath_length=1)
	
	
		self.multiresblock5 = Multiresblock(self.in_filters4,32*16)
		self.in_filters5 = int(32*16*self.alpha*0.167)+int(32*16*self.alpha*0.333)+int(32*16*self.alpha* 0.5)
	 
		# Add dropout after bottleneck
		self.dropout_bottleneck = torch.nn.Dropout2d(dropout_rate)
		
		# Decoder path
		self.upsample6 = torch.nn.ConvTranspose2d(self.in_filters5,32*8,kernel_size=(2,2),stride=(2,2))  
		self.concat_filters1 =  32*8 *2
		self.multiresblock6 = Multiresblock(self.concat_filters1,32*8)
		self.in_filters6 = int(32*8*self.alpha*0.167)+int(32*8*self.alpha*0.333)+int(32*8*self.alpha* 0.5)

		self.upsample7 = torch.nn.ConvTranspose2d(self.in_filters6,32*4,kernel_size=(2,2),stride=(2,2))  
		self.concat_filters2 =  32*4 *2
		self.multiresblock7 = Multiresblock(self.concat_filters2,32*4)
		self.in_filters7 = int(32*4*self.alpha*0.167)+int(32*4*self.alpha*0.333)+int(32*4*self.alpha* 0.5)
	
		self.upsample8 = torch.nn.ConvTranspose2d(self.in_filters7,32*2,kernel_size=(2,2),stride=(2,2))
		self.concat_filters3 =  32*2 *2
		self.multiresblock8 = Multiresblock(self.concat_filters3,32*2)
		self.in_filters8 = int(32*2*self.alpha*0.167)+int(32*2*self.alpha*0.333)+int(32*2*self.alpha* 0.5)

		self.upsample9 = torch.nn.ConvTranspose2d(self.in_filters8,32,kernel_size=(2,2),stride=(2,2))
		self.concat_filters4 =  32 *2
		self.multiresblock9 = Multiresblock(self.concat_filters4,32)
		self.in_filters9 = int(32*self.alpha*0.167)+int(32*self.alpha*0.333)+int(32*self.alpha* 0.5)

		# Add dropout before final layer
		self.dropout_before_final = torch.nn.Dropout2d(dropout_rate)
		
		# Update the final layer to produce the correct number of output channels
		self.conv_final = Conv2d_batchnorm(self.in_filters9, num_classes, kernel_size=(1, 1), activation='None')

	def forward(self,x : torch.Tensor)->torch.Tensor:

		x_multires1 = self.multiresblock1(x)
		x_pool1 = self.pool1(x_multires1)
		x_multires1 = self.respath1(x_multires1)
		
		x_multires2 = self.multiresblock2(x_pool1)
		x_pool2 = self.pool2(x_multires2)
		x_multires2 = self.respath2(x_multires2)

		x_multires3 = self.multiresblock3(x_pool2)
		x_pool3 = self.pool3(x_multires3)
		x_multires3 = self.respath3(x_multires3)

		x_multires4 = self.multiresblock4(x_pool3)
		x_pool4 = self.pool4(x_multires4)
		x_multires4 = self.respath4(x_multires4)

		x_multires5 = self.multiresblock5(x_pool4)
		
		# Apply dropout at bottleneck
		x_multires5 = self.dropout_bottleneck(x_multires5)

		up6 = torch.cat([self.upsample6(x_multires5),x_multires4],axis=1)
		x_multires6 = self.multiresblock6(up6)

		up7 = torch.cat([self.upsample7(x_multires6),x_multires3],axis=1)
		x_multires7 = self.multiresblock7(up7)

		up8 = torch.cat([self.upsample8(x_multires7),x_multires2],axis=1)
		x_multires8 = self.multiresblock8(up8)

		up9 = torch.cat([self.upsample9(x_multires8),x_multires1],axis=1)
		x_multires9 = self.multiresblock9(up9)
		
		# Apply dropout before final layer
		x_multires9 = self.dropout_before_final(x_multires9)

		out =  self.conv_final(x_multires9)  # Ensure the output has the correct number of channels
		return out

def dice_coef(y_true, y_pred):
    smooth = 1e-6  # To avoid division by zero
    # Flatten tensors while preserving batch and channel dimensions
    y_true_flat = y_true.view(y_true.size(0), -1)
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    
    intersection = (y_true_flat * y_pred_flat).sum(dim=1)
    union = y_true_flat.sum(dim=1) + y_pred_flat.sum(dim=1)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def jacard(y_true, y_pred):
    smooth = 1e-6  # To avoid division by zero
    # Flatten tensors while preserving batch and channel dimensions
    y_true_flat = y_true.view(y_true.size(0), -1)
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    
    intersection = (y_true_flat * y_pred_flat).sum(dim=1)
    union = y_true_flat.sum(dim=1) + y_pred_flat.sum(dim=1) - intersection
    
    jaccard = (intersection + smooth) / (union + smooth)
    return jaccard.mean()

def saveModel(model, model_dir='models', model_name='model.pth'):
    """
    Save the model architecture and weights.

    Arguments:
        model {torch.nn.Module} -- The PyTorch model to save.
        model_dir {str} -- Directory to save the model files (default: 'models').
    """
    import os
    import torch

    # Create the directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Save the model architecture
    model_arch_path = os.path.join(model_dir, 'model_architecture.pth')
    torch.save(model, model_arch_path)

    # Save the model weights
    model_weights_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_weights_path)

    print(f"Model architecture saved to {model_arch_path}")
    print(f"Model weights saved to {model_weights_path}")

def evaluateModel(model, X_test=None, Y_test=None, batch_size=2, device=None, val_loader=None):
    """
    Evaluate the model on test/validation data and compute metrics.

    Arguments:
        model {torch.nn.Module} -- The PyTorch model to evaluate.
        
    Keyword Arguments:
        X_test {torch.Tensor} -- Test input data (optional if val_loader provided).
        Y_test {torch.Tensor} -- Test ground truth labels (optional if val_loader provided).
        batch_size {int} -- Batch size for evaluation (default: 2).
        device {torch.device} -- The device to use (CPU or GPU) (optional).
        val_loader {DataLoader} -- Validation data loader (optional, will create from X_test/Y_test if not provided).
    
    Returns:
        tuple: (average_dice, average_jaccard)
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Create DataLoader if not provided
    if val_loader is None:
        if X_test is None or Y_test is None:
            raise ValueError("Either (X_test, Y_test) or val_loader must be provided")
        test_dataset = TensorDataset(X_test, Y_test)
        val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)

    model.eval()  # Set model to evaluation mode
    total_dice = 0
    total_jaccard = 0
    num_batches = 0

    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            # Move batch to device if specified
            if device is not None:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            # Forward pass
            Y_pred = model(X_batch)
            
            # Apply sigmoid to get probabilities (for BCEWithLogitsLoss output)
            Y_pred_prob = torch.sigmoid(Y_pred)
            # Threshold at 0.5 for binary segmentation
            Y_pred_binary = (Y_pred_prob >= 0.5).float()

            # Compute metrics on raw probabilities (more stable)
            dice = dice_coef(Y_batch, Y_pred_binary)
            jaccard = jacard(Y_batch, Y_pred_binary)

            total_dice += dice.item()
            total_jaccard += jaccard.item()
            num_batches += 1

    avg_dice = total_dice / num_batches
    avg_jaccard = total_jaccard / num_batches

    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    print(f"Average Jaccard Index: {avg_jaccard:.4f}")

    return avg_dice, avg_jaccard

def trainStep(model, X_train=None, Y_train=None, X_val=None, Y_val=None,
              train_loader=None, val_loader=None, epochs=50, batch_size=2, device='cuda', 
              learning_rate=1e-4, gradient_clip=1.0, weight_decay=0,
              num_workers=4, prefetch_factor=2,
              save_model=False, save_dir='models', verbose=False, log_dir=None,
              # additional run config for logging
              scale=False, scale_factor=0.5, data_limit=None, validation_split=0.1,
              input_channels=3, output_channels=4,
              train_augmentation=False, val_augmentation=False, repeat_factor=1, seed=42,
              # loss function configuration
              use_focal_loss=False, focal_alpha=0.25, focal_gamma=2.0,
              use_combined_loss=False, bce_weight=0.5, dice_weight=0.5,
              # learning rate scheduler configuration
              lr_scheduler_type='cosine', lr_step_size=30, lr_gamma=0.1, lr_patience=10,
              # early stopping configuration
              early_stopping_patience=15, early_stopping_min_delta=0.0,
              early_stopping_min_epochs=0,
              # augmentation curriculum configuration
              augmentation_curriculum='none', curriculum_start_epoch=0,
              curriculum_ramp_epochs=20, curriculum_max_aug_level=0.0,
              curriculum_base_strength='mild', curriculum_target_strength='moderate'):
    """
    Train the model for multiple epochs and evaluate after each epoch.

    Arguments:
        model {torch.nn.Module} -- The PyTorch model to train.
        
    Keyword Arguments:
        X_train {np.ndarray} -- Training input data (NumPy array, optional if train_loader provided).
        Y_train {np.ndarray} -- Training ground truth labels (NumPy array, optional if train_loader provided).
        X_val {np.ndarray} -- Validation input data (NumPy array, optional if val_loader provided).
        Y_val {np.ndarray} -- Validation ground truth labels (NumPy array, optional if val_loader provided).
        train_loader {DataLoader} -- Training data loader (optional, will create from X_train/Y_train if not provided).
        val_loader {DataLoader} -- Validation data loader (optional, will create from X_val/Y_val if not provided).
        epochs {int} -- Number of epochs to train (default: 50).
        batch_size {int} -- Batch size for training (default: 2).
        device {str/device} -- The device to use (CPU or GPU) (default: 'cuda').
        learning_rate {float} -- Initial learning rate (default: 1e-4)
        gradient_clip {float} -- Maximum gradient norm for clipping (default: 1.0). Set to 0 to disable.
        weight_decay {float} -- Weight decay (L2 regularization) (default: 0)
        num_workers {int} -- Number of worker processes for data loading (default: 4)
        prefetch_factor {int} -- Number of batches loaded in advance by each worker (default: 2)
        save_model {bool} -- Whether to save model checkpoints (default: False)
        save_dir {str} -- Directory to save model checkpoints (default: 'models')
        verbose {bool} -- Enable verbose logging (default: False)
        log_dir {str} -- Directory for TensorBoard logs (default: None)
        use_focal_loss {bool} -- Use Focal Loss instead of BCE (default: False, recommended for sparse/imbalanced datasets)
        focal_alpha {float} -- Focal Loss alpha parameter (default: 0.25)
        focal_gamma {float} -- Focal Loss gamma parameter (default: 2.0)
        use_combined_loss {bool} -- Use combined BCE/Focal + Dice Loss (default: False)
        bce_weight {float} -- Weight for BCE/Focal loss in combined loss (default: 0.5)
        dice_weight {float} -- Weight for Dice loss in combined loss (default: 0.5)
        early_stopping_patience {int} -- Epochs to wait for val Dice improvement before stopping. Set 0 to disable.
        early_stopping_min_delta {float} -- Minimum val Dice improvement to reset early stopping.
        early_stopping_min_epochs {int} -- Do not early stop before this epoch. Useful for augmentation curricula.
        augmentation_curriculum {str} -- 'none', 'linear', or 'cosine'.
        curriculum_start_epoch {int} -- Epoch where augmentation strength starts ramping up.
        curriculum_ramp_epochs {int} -- Number of epochs used to ramp augmentation strength.
        curriculum_max_aug_level {float} -- Max interpolation level toward curriculum_target_strength.
        curriculum_base_strength {str} -- Base augmentation profile used before/alongside the curriculum.
        curriculum_target_strength {str} -- Target augmentation profile used by the curriculum.
    
    Returns:
        dict: Training history containing loss and metrics
    """
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # Move model to the specified device
    device = torch.device(device)
    model.to(device)

    # Create DataLoader if not provided
    if train_loader is None:
        if X_train is None or Y_train is None:
            raise ValueError("Either (X_train, Y_train) or train_loader must be provided")
        
        # Convert numpy arrays to tensors if needed
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
        if not isinstance(Y_train, torch.Tensor):
            Y_train = torch.tensor(Y_train, dtype=torch.float32).permute(0, 3, 1, 2)
        
        train_dataset = TensorDataset(X_train, Y_train)
        
        # Handle prefetch_factor for num_workers=0
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'pin_memory': True,
            'persistent_workers': False
        }
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = prefetch_factor
        
        train_loader = DataLoader(train_dataset, **loader_kwargs)
    
    if val_loader is None:
        if X_val is None or Y_val is None:
            raise ValueError("Either (X_val, Y_val) or val_loader must be provided")
        
        # Convert numpy arrays to tensors if needed
        if not isinstance(X_val, torch.Tensor):
            X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
        if not isinstance(Y_val, torch.Tensor):
            Y_val = torch.tensor(Y_val, dtype=torch.float32).permute(0, 3, 1, 2)
        
        val_dataset = TensorDataset(X_val, Y_val)
        
        # Handle prefetch_factor for num_workers=0
        val_loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'pin_memory': True,
            'persistent_workers': False
        }
        if num_workers > 0 and prefetch_factor is not None:
            val_loader_kwargs['prefetch_factor'] = prefetch_factor
        
        val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    
    # Define loss function and optimizer
    # Use Focal Loss or Combined Loss for sparse/imbalanced datasets
    if use_combined_loss:
        print(f"\nUsing Combined Loss (BCE/Focal + Dice)")
        print(f"  BCE/Focal weight: {bce_weight:.1f}")
        print(f"  Dice weight: {dice_weight:.1f}")
        if use_focal_loss:
            print(f"  Using Focal Loss with alpha={focal_alpha}, gamma={focal_gamma}")
        criterion = CombinedLoss(
            bce_weight=bce_weight, 
            dice_weight=dice_weight,
            use_focal=use_focal_loss,
            alpha=focal_alpha,
            gamma=focal_gamma
        )
    elif use_focal_loss:
        print(f"\nUsing Focal Loss (recommended for sparse/imbalanced datasets)")
        print(f"  Alpha: {focal_alpha}, Gamma: {focal_gamma}")
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        print(f"\nUsing BCEWithLogitsLoss")
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler - Support different types for flexibility
    if hasattr(train_loader, '__len__'):
        steps_per_epoch = len(train_loader)
    else:
        # Estimate steps per epoch if train_loader doesn't support len()
        steps_per_epoch = max(1, len(X_train) // batch_size if X_train is not None else 100)
    
    total_steps = epochs * steps_per_epoch
    
    if isinstance(learning_rate, (int, float)):
        base_lr = learning_rate
    else:
        base_lr = 1e-4  # default fallback
    
    # Learning rate scheduler - Support multiple types for different scenarios
    if lr_scheduler_type == 'cosine':
        # Cosine annealing - good for stable convergence on sparse data
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)
    elif lr_scheduler_type == 'step':
        # Step decay - simple and effective, reduces LR by factor every N epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    elif lr_scheduler_type == 'exponential':
        # Exponential decay - smooth continuous decay
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    elif lr_scheduler_type == 'plateau':
        # Reduce on plateau - adaptive, reduces LR when metric stops improving
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=lr_gamma, patience=lr_patience, verbose=True
        )
    else:
        # Default to cosine annealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)

    # Setup TensorBoard writer if log_dir is provided
    writer = None
    if log_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=log_dir)
            print(f"\n✓ TensorBoard logging enabled at: {log_dir}")
            
            # Record comprehensive hyperparameters and configuration
            import json
            
            # Construct complete hyperparameter dictionary
            hparams = {
                # Training hyperparameters
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'gradient_clip': gradient_clip,
                'weight_decay': weight_decay,
                
                # Scheduler parameters
                'lr_scheduler_type': lr_scheduler_type,
                'lr_step_size': lr_step_size,
                'lr_gamma': lr_gamma,
                'lr_patience': lr_patience,
                'early_stopping_patience': early_stopping_patience,
                'early_stopping_min_delta': early_stopping_min_delta,
                'early_stopping_min_epochs': early_stopping_min_epochs,
                'augmentation_curriculum': augmentation_curriculum,
                'curriculum_start_epoch': curriculum_start_epoch,
                'curriculum_ramp_epochs': curriculum_ramp_epochs,
                'curriculum_max_aug_level': curriculum_max_aug_level,
                'curriculum_base_strength': curriculum_base_strength,
                'curriculum_target_strength': curriculum_target_strength,
                
                # DataLoader parameters
                'num_workers': num_workers,
                'prefetch_factor': prefetch_factor,
                
                # Model architecture
                'input_channels': input_channels,
                'output_channels': output_channels,
                
                # Data configuration
                'data_limit': data_limit if data_limit else -1,  # -1 means all data
                'validation_split': validation_split,
                'scale': bool(scale),
                'scale_factor': scale_factor,
                'train_augmentation': train_augmentation,
                'val_augmentation': val_augmentation,
                'repeat_factor': repeat_factor,
                'seed': seed,
                
                # Loss function configuration
                'use_focal_loss': use_focal_loss,
                'focal_alpha': focal_alpha,
                'focal_gamma': focal_gamma,
                'use_combined_loss': use_combined_loss,
                'bce_weight': bce_weight,
                'dice_weight': dice_weight,
                
                # Device info
                'device': str(device),
            }
            
            # Add readable config text to TensorBoard
            config_text = json.dumps(hparams, indent=2)
            writer.add_text('Training_Config', config_text)
            
            # Register hyperparameters for comparison
            filtered_hparams = {
                k: v for k, v in hparams.items() 
                if isinstance(v, (int, float, str, bool)) and not k.startswith('_')
            }
            
            # Add placeholder metric for hparams registration
            writer.add_hparams(
                filtered_hparams,
                {'hparam/placeholder': 0.0}
            )
            
            # Flush immediately to ensure data is written
            writer.flush()
            
            print(f"  ✓ Hyperparameters recorded")
            
        except ImportError:
            print("\n⚠ WARNING: TensorBoard not installed. Install with: pip install tensorboard")
            writer = None
        except Exception as e:
            print(f"\n⚠ WARNING: Failed to initialize TensorBoard: {e}")
            writer = None

    # Initialize current_lr before training loop
    current_lr = learning_rate

    # Training loop
    history = {
        'loss': [],
        'dice': [],
        'jaccard': [],
        'val_dice': [],
        'val_jaccard': [],
        'augmentation_schedule_level': []
    }
    best_val_dice = float('-inf')
    best_val_jaccard = 0.0
    best_epoch = 0
    avg_val_losses = []
    epochs_without_improvement = 0
    early_stopping_enabled = early_stopping_patience is not None and early_stopping_patience > 0

    def get_curriculum_level(epoch_number):
        if augmentation_curriculum == 'none':
            return 0.0
        if epoch_number < curriculum_start_epoch:
            return 0.0
        if curriculum_ramp_epochs <= 0:
            return float(curriculum_max_aug_level)
        progress = (epoch_number - curriculum_start_epoch + 1) / float(curriculum_ramp_epochs)
        progress = max(0.0, min(1.0, progress))
        if augmentation_curriculum == 'cosine':
            import math
            progress = 0.5 - 0.5 * math.cos(math.pi * progress)
        return float(curriculum_max_aug_level) * progress

    def update_loader_augmentation(epoch_number):
        schedule_level = get_curriculum_level(epoch_number)
        dataset = getattr(train_loader, 'dataset', None)
        if hasattr(dataset, 'set_augmentation_mix'):
            dataset.set_augmentation_mix(
                base_strength=curriculum_base_strength,
                strong_prob=0.0,
                strong_strength=curriculum_target_strength,
                schedule_level=schedule_level
            )
        return schedule_level
    
    for epoch in range(epochs):
        epoch_number = epoch + 1
        augmentation_schedule_level = update_loader_augmentation(epoch_number)
        model.train()  # Set model to training mode
        running_loss = 0.0
        total_dice = 0.0
        total_jaccard = 0.0
        num_batches = 0

        for X_batch, Y_batch in train_loader:
            # Move batch to device if specified
            if device is not None:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)

            # Backward pass and optimize
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            # Compute metrics on raw probabilities (more stable)
            Y_pred_prob = torch.sigmoid(Y_pred)
            Y_pred_binary = (Y_pred_prob >= 0.5).float()
            dice = dice_coef(Y_batch, Y_pred_binary)
            jaccard = jacard(Y_batch, Y_pred_binary)

            running_loss += loss.item()
            total_dice += dice.item()
            total_jaccard += jaccard.item()
            num_batches += 1

        epoch_loss = running_loss / num_batches
        epoch_dice = total_dice / num_batches
        epoch_jaccard = total_jaccard / num_batches

        # Evaluate on validation set
        val_dice, val_jaccard = evaluateModel(model, val_loader=val_loader, device=device)

        # Update history
        history['loss'].append(epoch_loss)
        history['dice'].append(epoch_dice)
        history['jaccard'].append(epoch_jaccard)
        history['val_dice'].append(val_dice)
        history['val_jaccard'].append(val_jaccard)
        history['augmentation_schedule_level'].append(augmentation_schedule_level)

        # Track best validation Dice and save the best checkpoint.
        is_best = val_dice > best_val_dice + early_stopping_min_delta
        if is_best:
            best_val_dice = val_dice
            best_val_jaccard = val_jaccard
            best_epoch = epoch + 1
            epochs_without_improvement = 0  # Reset counter when improvement occurs
            if save_model:
                saveModel(model, model_dir=save_dir, model_name='best_model.pth')
                print(f"  ✓ New best model saved by val Dice: {best_val_dice:.4f} (epoch {best_epoch})")
        else:
            epochs_without_improvement += 1

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary with improved formatting (always show, not just in verbose mode)
        print(f"\nEpoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        print(f"Average Dice Coefficient: {epoch_dice:.4f}")
        print(f"Average Jaccard Index: {epoch_jaccard:.4f}")
        print(f"  Current learning rate: {current_lr:.6f}")
        if augmentation_curriculum != 'none':
            print(f"  Augmentation curriculum: {augmentation_curriculum}, level={augmentation_schedule_level:.3f}, target_profile={curriculum_target_strength}")
        print(f"  Validation Dice: {val_dice:.4f}, Jaccard: {val_jaccard:.4f}")
        print(f"  Best Validation Dice: {best_val_dice:.4f} (epoch {best_epoch})")
        if early_stopping_enabled:
            print(f"  Patience Counter: {epochs_without_improvement}/{early_stopping_patience}")
        
        # Log to TensorBoard
        if writer is not None:
            try:
                writer.add_scalar('Loss/train', epoch_loss, epoch+1)
                # Use actual validation loss calculation for more accurate tracking
                with torch.no_grad():
                    model.eval()
                    val_running_loss = 0.0
                    val_batches = 0
                    for X_val_batch, Y_val_batch in val_loader:
                        if device is not None:
                            X_val_batch, Y_val_batch = X_val_batch.to(device), Y_val_batch.to(device)
                        Y_val_pred = model(X_val_batch)
                        val_loss = criterion(Y_val_pred, Y_val_batch)
                        val_running_loss += val_loss.item()
                        val_batches += 1
                    avg_val_loss = val_running_loss / val_batches
                    avg_val_losses.append(avg_val_loss)
                    writer.add_scalar('Loss/validation', avg_val_loss, epoch+1)
                model.train()  # Switch back to training mode
                
                writer.add_scalar('Metrics/dice', epoch_dice, epoch+1)
                writer.add_scalar('Metrics/jaccard', epoch_jaccard, epoch+1)
                writer.add_scalar('Metrics/val_dice', val_dice, epoch+1)
                writer.add_scalar('Metrics/val_jaccard', val_jaccard, epoch+1)
                writer.add_scalar('Metrics/best_val_dice', best_val_dice, epoch+1)
                writer.add_scalar('Metrics/best_epoch', best_epoch, epoch+1)
                writer.add_scalar('Learning_rate', current_lr, epoch+1)
                writer.add_scalar('Augmentation/schedule_level', augmentation_schedule_level, epoch+1)
                writer.flush()  # Force flush to prevent memory buildup
            except Exception as e:
                if epoch == 0:  # Only warn once
                    print(f"⚠ WARNING: Failed to write to TensorBoard: {e}")

        # Save model checkpoint
        if save_model and (epoch + 1) % 10 == 0:  # Save every 10 epochs
            saveModel(model, model_dir=save_dir, model_name=f'model_epoch_{epoch+1}.pth')

        # Update learning rate based on scheduler type
        if lr_scheduler_type == 'plateau':
            # For ReduceLROnPlateau, we need to pass the monitored metric
            scheduler.step(val_dice)  # Pass validation dice as the metric to monitor
        else:
            # For other schedulers, just call step()
            scheduler.step()

        # Early stopping check after logging/checkpointing this epoch.
        can_early_stop = (epoch + 1) >= early_stopping_min_epochs
        if early_stopping_enabled and can_early_stop and epochs_without_improvement >= early_stopping_patience:
            print(f"\n⚠️  Early stopping triggered after {early_stopping_patience} epochs without val Dice improvement.")
            print(f"Best validation Dice: {best_val_dice:.4f} at epoch {best_epoch}")
            break  # Exit the training loop

        # AGGRESSIVE memory cleanup after EVERY epoch (CRITICAL FIX - Enhanced)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        import gc
        gc.collect()  # FORCE Python garbage collection

        # Monitor memory every 5 epochs (more frequent monitoring)
        if (epoch + 1) % 5 == 0 and device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            reserved = torch.cuda.memory_reserved(device) / 1024**2
            print(f"  📊 GPU Memory Status: Allocated={allocated:.0f}MB, Reserved={reserved:.0f}MB")
            
            # Warning if memory usage is high
            if allocated > 6000:  # 6GB threshold
                print(f"  ⚠ WARNING: High GPU memory usage detected. Consider reducing batch_size.")

    print("\nTraining complete.")
    print(f"Final Dice: {history['val_dice'][-1]:.4f}")
    print(f"Final Jaccard: {history['val_jaccard'][-1]:.4f}")
    print(f"Best Validation Dice: {best_val_dice:.4f} at epoch {best_epoch}")
    
    # Close TensorBoard writer and log final hparams/metrics
    if writer is not None:
        try:
            # Update hparams with FINAL metrics (CRITICAL FIX)
            # This should be done ONCE at the end, replacing the placeholder
            final_metrics = {
                'final/val_dice': float(best_val_dice),
                'final/val_jaccard': float(best_val_jaccard),
                'final/best_epoch': float(best_epoch),
                'final/train_loss': float(history['loss'][-1]) if history['loss'] else 0.0,
                'final/val_loss': float(avg_val_losses[-1]) if avg_val_losses else 0.0
            }
            
            # Use COMPLETE hparams (all training parameters)
            hparams = {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'gradient_clip': gradient_clip,
                'weight_decay': weight_decay,
                'lr_scheduler_type': lr_scheduler_type,
                'lr_step_size': lr_step_size,
                'lr_gamma': lr_gamma,
                'lr_patience': lr_patience,
                'early_stopping_patience': early_stopping_patience,
                'early_stopping_min_delta': early_stopping_min_delta,
                'early_stopping_min_epochs': early_stopping_min_epochs,
                'augmentation_curriculum': augmentation_curriculum,
                'curriculum_start_epoch': curriculum_start_epoch,
                'curriculum_ramp_epochs': curriculum_ramp_epochs,
                'curriculum_max_aug_level': curriculum_max_aug_level,
                'curriculum_base_strength': curriculum_base_strength,
                'curriculum_target_strength': curriculum_target_strength,
                'num_workers': num_workers,
                'prefetch_factor': prefetch_factor,
                'save_model': save_model,
                'scale': bool(scale),
                'scale_factor': scale_factor,
                'data_limit': data_limit,
                'validation_split': validation_split,
                'input_channels': input_channels,
                'output_channels': output_channels,
                'train_augmentation': train_augmentation,
                'val_augmentation': val_augmentation,
                'repeat_factor': repeat_factor,
                'seed': seed,
                'device': str(device)
            }
            
            # Log final metrics to replace placeholder
            writer.add_hparams(hparams, final_metrics)
            writer.flush()
            
            print(f"  ✓ Final metrics recorded to TensorBoard")
            
        except Exception as e:
            print(f"  ⚠ Warning: Failed to record final metrics: {e}")
        
        writer.close()  # Close the writer to free resources

    # Save training history to the same directory as TensorBoard logs if available
    import numpy as np
    import os
    history_save_dir = log_dir if log_dir is not None else 'runs/history'
    os.makedirs(history_save_dir, exist_ok=True)
    history_path = os.path.join(history_save_dir, 'training_history.npy')
    np.save(history_path, history)
    print(f"Training history saved to '{history_path}'")

    return history
