import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path


def unnormalize(img, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    img = img * std + mean
    return img

def load_moco_v2_model(checkpoint_path):
    model = models.resnet50()

    model.fc = torch.nn.Identity()

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state_dict = checkpoint['state_dict']


    new_state_dict = {k.replace('module.encoder_q.', ''): v for k, v in state_dict.items() if 'encoder_q' in k}


    model.load_state_dict(new_state_dict, strict=False)

    return model

def get_loaders(nyu_dataset, batch_size=16, test_size=0.2, random_state=42):
    indices = list(range(len(nyu_dataset)))
    train_indices, eval_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    train_dataset = Subset(nyu_dataset, train_indices)
    eval_dataset = Subset(nyu_dataset, eval_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, eval_loader

def train(train_loader, eval_loader, model, optimizer, criterion, num_epochs, device):
    train_losses = []
    train_rmse_scores = []
    eval_losses = []
    eval_rmse_scores = []
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_train_loss = 0.0
        total_train_rmse = 0.0
        
        for batch in train_loader:
            images, depths = batch
            images = images.to(device)
            depths = depths.to(device)
            
            preds = model(images)
            loss = criterion(preds, depths)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            with torch.no_grad():
                mse = torch.mean((preds - depths) ** 2)
                rmse = torch.sqrt(mse)
                total_train_rmse += rmse.item()
                
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_rmse = total_train_rmse / len(train_loader)
        
        model.eval()
        total_eval_loss = 0.0
        total_eval_rmse = 0.0
        
        with torch.no_grad():
            for batch in eval_loader:
                images, depths = batch
                images = images.to(device)
                depths = depths.to(device)
                
                preds = model(images)
                loss = criterion(preds, depths)
                
                total_eval_loss += loss.item()
                mse = torch.mean((preds - depths) ** 2)
                rmse = torch.sqrt(mse)
                total_eval_rmse += rmse.item()
        
        # Average validation metrics
        avg_eval_loss = total_eval_loss / len(eval_loader)
        avg_eval_rmse = total_eval_rmse / len(eval_loader)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        train_rmse_scores.append(avg_train_rmse)
        eval_losses.append(avg_eval_loss)
        eval_rmse_scores.append(avg_eval_rmse)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_eval_loss:.4f} | RMSE: {avg_eval_rmse:.4f}')
    
    return {
        'train_losses': train_losses,
        'train_rmse': train_rmse_scores,
        'eval_losses': eval_losses,
        'eval_rmse': eval_rmse_scores
    }

def plot_final_metrics(metrics, model_name="MoCo-v2 Model"):
    """
    Plot final training metrics history.

    Args:
        metrics (dict): Dictionary containing training history
        model_name (str): Name of the model for plot titles
    """
    plt.style.use('seaborn-v0_8-dark')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(metrics['train_losses']) + 1)

    ax1.plot(epochs, metrics['train_losses'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, metrics['eval_losses'], 'g-', label='Eval', linewidth=2)
    ax1.set_title(f'{model_name} - L1 Loss Progress', pad=10)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, metrics['train_rmse'], 'r-', label='Train', linewidth=2)
    ax2.plot(epochs, metrics['eval_rmse'], 'm-', label='Eval', linewidth=2)
    ax2.set_title(f'{model_name} - RMSE Progress', pad=10)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    loss_ratio = [e/t for e, t in zip(metrics['eval_losses'], metrics['train_losses'])]
    ax3.plot(epochs, loss_ratio, 'y-', label='Eval/Train Ratio', linewidth=2)
    ax3.set_title(f'{model_name} - Loss Ratio (Eval/Train)', pad=10)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Ratio')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    best_train_loss = min(metrics['train_losses'])
    best_eval_loss = min(metrics['eval_losses'])
    best_train_rmse = min(metrics['train_rmse'])
    best_eval_rmse = min(metrics['eval_rmse'])

    metrics_text = f'Best Metrics:\n\n'
    metrics_text += f'Train Loss: {best_train_loss:.4f}\n'
    metrics_text += f'Eval Loss: {best_eval_loss:.4f}\n\n'
    metrics_text += f'Train RMSE: {best_train_rmse:.4f}\n'
    metrics_text += f'Eval RMSE: {best_eval_rmse:.4f}\n\n'
    metrics_text += f'Final Loss Ratio: {loss_ratio[-1]:.4f}'

    ax4.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=10)
    ax4.set_title(f'{model_name} - Best Metrics', pad=10)
    ax4.axis('off')

    plt.tight_layout()
    plt.show()

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def readable_number(num):
    num_str = str(num)[::-1]
    res = ''
    i_prev = 0
    for i in range(3, len(num_str), 3):
        res += num_str[i_prev:i] + ','
        i_prev = i
    if i_prev < len(num_str):
        res += num_str[i_prev:]
    return res[::-1]

def log(writer, metrics, epoch):
    writer.add_scalars('loss', {'train': metrics['loss_train'], 'test': metrics['loss_test']}, epoch)
    writer.add_scalars('accuracy', {'train': metrics['accuracy_train'], 'test': metrics['accuracy_test']}, epoch)
    writer.flush()

def save_checkpoint(state, path, epoch, test_loss):
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(state, f'{path}/{epoch}_valloss={test_loss:.3f}.pt')

def get_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return total, trainable

def print_parameters(model):
    total, trainable = get_parameters(model)
    print(f'model initialized with trainable params: {readable_number(trainable)} || total params: {readable_number(total)} || trainable%: {trainable/total * 100:.3f}')

def train_fp16(train_loader, eval_loader, model, optimizer, criterion, num_epochs, device, scheduler=None, scaler=None, writer=None, freq_save=10, save_path=None):
    train_losses = []
    train_rmse_scores = []
    eval_losses = []
    eval_rmse_scores = []
    
    for epoch in tqdm(range(num_epochs)):
        # Training Phase
        model.train()
        total_train_loss = 0.0
        total_train_rmse = 0.0
        
        for batch in train_loader:
            images, depths = batch
            images = images.to(device)
            depths = depths.to(device)
            
            # FP16 training
            with torch.amp.autocast('cuda'):
                preds = model(images)
                loss = criterion(preds, depths)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            with torch.no_grad():
                mse = torch.mean((preds - depths) ** 2)
                rmse = torch.sqrt(mse)
                total_train_rmse += rmse.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_rmse = total_train_rmse / len(train_loader)
        
        # Validation Phase
        model.eval()
        total_eval_loss = 0.0
        total_eval_rmse = 0.0
        
        with torch.no_grad():
            for batch in eval_loader:
                images, depths = batch
                images = images.to(device)
                depths = depths.to(device)
                
                with torch.amp.autocast('cuda'):
                    preds = model(images)
                    loss = criterion(preds, depths)
                
                total_eval_loss += loss.item()
                mse = torch.mean((preds - depths) ** 2)
                rmse = torch.sqrt(mse)
                total_eval_rmse += rmse.item()
        
        # Average validation metrics
        avg_eval_loss = total_eval_loss / len(eval_loader)
        avg_eval_rmse = total_eval_rmse / len(eval_loader)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        train_rmse_scores.append(avg_train_rmse)
        eval_losses.append(avg_eval_loss)
        eval_rmse_scores.append(avg_eval_rmse)
        
        # Optional logging and checkpointing
        if writer is not None:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/val', avg_eval_loss, epoch)
            writer.add_scalar('RMSE/val', avg_eval_rmse, epoch)
        
        if scheduler is not None:
            scheduler.step()
            
        if save_path and epoch % freq_save == 0:
            save_checkpoint(
                state=model.state_dict(),
                path=save_path,
                epoch=epoch,
                test_loss=avg_eval_loss
            )
        
        print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_eval_loss:.4f} | RMSE: {avg_eval_rmse:.4f}')
    
    return {
        'train_losses': train_losses,
        'train_rmse': train_rmse_scores,
        'eval_losses': eval_losses,
        'eval_rmse': eval_rmse_scores
    }

