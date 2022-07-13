import torch.utils.data
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image


# --------------- check image -------------------------------------------------------
def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False


# ------------------   dataset -------------------------------------------------------
train_data_path = "../train/"

transforms = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整为统一大小 H、W
    transforms.ToTensor(),  # 将图像转成tensor
    # 正规化，服从高斯分布，保证值都在0~1之间，防止做乘积的时候值过大（梯度爆炸 exploding gradient)
    transforms.Normalize(mean=[0.485, 0.465, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 训练数据集，更新模型
train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transforms, is_valid_file=check_image)

# 验证数据集，调整超参数
val_data_path = "../val/"
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=transforms, is_valid_file=check_image)

# 测试数据集，评价模型
test_data_path = "../test/"
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transforms, is_valid_file=check_image)

# ------------------  data loader -------------------------------------------------------
# 分批，每批大小为64个图像
batch_size = 8192
train_data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size)
val_data_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size)


# ------------------   Simple NET -------------------------------------------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 三个全连接层， keras中叫Dense， 这里叫Linear
        # 12288 = 通道*H*W = 3*64*64
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        # 将x转成一维的张量，才能输入全连接层
        # 设置打印的元素数量限制
        torch.set_printoptions(threshold=2000000)
        print("x.shape={}".format(x.shape))
        # print(x)
        # 每个图像大小12288，一共有多少个图像，不知道，你帮我算吧，是这个意思
        x = x.view(-1, 12288)
        # 按顺序应用全连接层 full connection (fc1,fc2,fc3)，和激活函数，F.relu和softmax函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 最后一个原样输出，是因为对于交叉熵（cross entropy Loss)来讲，最后一个操作是结合了softmax的
        # 叫 softmax with loss
        x = self.fc3(x)
        return x


simplenet = SimpleNet()

# ----------------------------  优化  optimizer -------------------------------------------------------
# parameters：将要被更新的网络权重
# 学习率增大，变成猫了，错了
# 学习率减少，还是能识别出来鱼，太小，就会出现识别为猫的情况
optimizer_1 = optim.Adam(simplenet.parameters(), lr=0.001)
# 采用SGD也能识另出来是fish
# optimizer_1 = optim.SGD(simplenet.parameters(), lr=0.001)


# ---------- 迁移到GPU --------------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
simplenet.to(device)


# --------------- train ------------------------------------------------------------------------------
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    try:
        for epoch in range(epochs):
            training_loss = 0.0
            valid_loss = 0.0
            model.train()
            for batch in train_loader:
                print("bach.size=:{}".format(len(batch[0])))
                # 为每个batch初始化grad为0
                optimizer.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(device)
                # 如可以，迁到GPU
                targets = targets.to(device)
                # 训练
                output = model(inputs)
                # 计算损失
                loss = loss_fn(output, targets)
                # 计算grad
                loss.backward()
                # 更新权重等参数
                optimizer.step()
                # 训练损失计算
                training_loss += loss.data.item() * inputs.size(0)
            training_loss /= len(train_loader.dataset)

            # 以下是验证过程
            model.eval()
            num_correct = 0
            num_examples = 0
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                output = model(inputs)
                # 迁移到GPU
                targets = targets.to(device)
                # loss_fn（参数）
                loss = loss_fn(output, targets)
                # valid set 的损失
                valid_loss += loss.data.item() * inputs.size(0)
                # 取出targets中最大的元素的值的
                correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], targets).view(-1)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
            valid_loss /= len(val_loader.dataset)

            print('==> 训练模型： Epoch:{}, Training Loss:{:.2f}, Validation Loss:{:.2f}, accuracy:{:.2f}'.format(
                epoch, training_loss, valid_loss, num_correct / num_examples
            ))
    except Exception as e:
        print("ERROR: {}".format(e))


# 训练模型
train(simplenet, optimizer_1, torch.nn.CrossEntropyLoss(), train_data_loader, val_data_loader, epochs=5, device=device)

# ------------------------ making predictions -----------------------------------------------------------
labels = ["cat", "fish"]

imgPath = "../val/fish/100_1422.JPG"
img = Image.open(imgPath)
# 移到GPU
img = transforms(img).to(device)
# 增加一个批次的维度
img = img.unsqueeze(0)

prediction = F.softmax(simplenet(img))
prediction = prediction.argmax()
print("预测图片：{} 是 {}".format(imgPath, labels[prediction]))

# ------------------------- 保存模型 ---------------------------------------------------------------------
torch.save(simplenet, "/tmp/simplenet")
simplenet = torch.load("/tmp/simplenet")

torch.save(simplenet.state_dict(), "/tmp/simplenet")
simplenet = SimpleNet()
simplenet_state_dict = torch.load("/tmp/simplenet")
simplenet.load_state_dict(simplenet_state_dict)
