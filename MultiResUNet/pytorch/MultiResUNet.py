import torch
import torch.nn.functional as F


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
	
	Returns:
		[keras model] -- MultiResUNet model
	'''
	def __init__(self, input_channels, num_classes, alpha=1.67):
		super().__init__()
		
		self.alpha = alpha
		
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

		up6 = torch.cat([self.upsample6(x_multires5),x_multires4],axis=1)
		x_multires6 = self.multiresblock6(up6)

		up7 = torch.cat([self.upsample7(x_multires6),x_multires3],axis=1)
		x_multires7 = self.multiresblock7(up7)

		up8 = torch.cat([self.upsample8(x_multires7),x_multires2],axis=1)
		x_multires8 = self.multiresblock8(up8)

		up9 = torch.cat([self.upsample9(x_multires8),x_multires1],axis=1)
		x_multires9 = self.multiresblock9(up9)

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

def saveModel(model, model_dir='models'):
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
    model_weights_path = os.path.join(model_dir, 'model_weights.pth')
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
              save_model=False, save_dir='models', verbose=False, log_dir=None):
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
    
    Returns:
        dict: Training history containing loss and metrics
    """
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn
    import torch.optim as optim
    import os
    import gc
    
    # Setup TensorBoard writer if log_dir is provided
    if log_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=log_dir)
            print(f"✓ TensorBoard logging enabled at: {log_dir}")
        except ImportError:
            print("⚠ WARNING: tensorboard not installed. Installing with: pip install tensorboard")
            print("  Skipping TensorBoard logging...")
            writer = None
            log_dir = None
    else:
        writer = None

    # Convert device string to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)
    
    # Create DataLoaders if not provided
    # Priority: 1) Provided loaders, 2) NumPy arrays
    if train_loader is None:
        if X_train is None or Y_train is None:
            raise ValueError("Either (X_train, Y_train) or train_loader must be provided")
        
        # Convert NumPy arrays to PyTorch tensors
        print(f"Converting training data to tensors...")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).permute(0, 3, 1, 2)
        
        # Clear NumPy arrays from memory
        del X_train, Y_train
        gc.collect()
        
        print(f"Training tensor shape: {X_train_tensor.shape}")
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        # Use configurable workers for data loading
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=num_workers, pin_memory=True, 
                                 prefetch_factor=prefetch_factor, persistent_workers=True)

    if val_loader is None:
        if X_val is None or Y_val is None:
            raise ValueError("Either (X_val, Y_val) or val_loader must be provided")
        
        print(f"Converting validation data to tensors...")
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2)
        Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).permute(0, 3, 1, 2)
        
        # Clear NumPy arrays from memory
        del X_val, Y_val
        gc.collect()
        
        print(f"Validation tensor shape: {X_val_tensor.shape}")
        val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
        # Use configurable workers for validation data loading
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True, 
                               prefetch_factor=prefetch_factor, persistent_workers=True)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training history
    history = {'train_loss': [], 'val_dice': [], 'val_jaccard': [], 'val_loss': []}
    
    # Best model tracking
    best_val_dice = 0.0
    best_model_state = None
    
    # Initialize current_lr before training loop
    current_lr = learning_rate

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        batch_count = 0

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            Y_pred = model(X_batch)

            # Compute loss
            loss = criterion(Y_pred, Y_batch)
            loss.backward()  # Backpropagation
            
            # Gradient clipping to prevent exploding gradients
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            
            optimizer.step()  # Update weights

            running_loss += loss.item()
            batch_count += 1
            
            # Memory cleanup: Release batch tensors
            del Y_pred, loss
            if (batch_count % 10 == 0) and (device.type == 'cuda'):
                torch.cuda.empty_cache()

        avg_loss = running_loss / batch_count
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Evaluate on validation data (pass device for GPU acceleration)
        avg_dice, avg_jaccard = evaluateModel(model, None, None, batch_size, device=device, val_loader=val_loader)
        
        # Calculate validation loss
        val_loss = 0.0
        val_batch_count = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                Y_pred = model(X_batch)
                val_loss += criterion(Y_pred, Y_batch).item()
                val_batch_count += 1
                
                # Cleanup validation tensors
                del Y_pred
        avg_val_loss = val_loss / val_batch_count
        
        # Store history
        history['train_loss'].append(avg_loss)
        history['val_dice'].append(avg_dice)
        history['val_jaccard'].append(avg_jaccard)
        history['val_loss'].append(avg_val_loss)
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train', avg_loss, epoch+1)
            writer.add_scalar('Loss/validation', avg_val_loss, epoch+1)
            writer.add_scalar('Metrics/dice', avg_dice, epoch+1)
            writer.add_scalar('Metrics/jaccard', avg_jaccard, epoch+1)
            writer.add_scalar('Learning_rate', current_lr, epoch+1)
        
        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Print learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Current learning rate: {current_lr:.6f}")
        print(f"  Validation Dice: {avg_dice:.4f}, Jaccard: {avg_jaccard:.4f}")
        
        # Save best model
        if save_model and avg_dice > best_val_dice:
            best_val_dice = avg_dice
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': avg_dice,
                'val_jaccard': avg_jaccard,
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, 'best_model_checkpoint.pth'))
            print(f"  ✓ New best model saved! (Dice: {avg_dice:.4f})")
        
        # Early stopping check
        if epoch > 10 and avg_dice < 0.01:
            print(f"\n⚠ WARNING: Dice coefficient is very low after {epoch+1} epochs.")
            print("Consider checking data quality or adjusting hyperparameters.")
            if verbose:
                print("Debug info:")
                print(f"  Training loss: {avg_loss:.4f}")
                print(f"  Validation loss: {avg_val_loss:.4f}")
        
        # Periodic memory cleanup (every 10 epochs)
        if (epoch + 1) % 10 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            print(f"  ✓ Memory cleanup completed")

    print("\nTraining complete.")
    print(f"Final Dice: {history['val_dice'][-1]:.4f}")
    print(f"Final Jaccard: {history['val_jaccard'][-1]:.4f}")
    print(f"Best Validation Dice: {best_val_dice:.4f}")
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print(f"✓ TensorBoard logs saved to: {log_dir}")
        print(f"  Run 'tensorboard --logdir {log_dir}' to view training curves")
    
    # Save training history
    import numpy as np
    np.save('training_history.npy', history)
    print("Training history saved to 'training_history.npy'")
