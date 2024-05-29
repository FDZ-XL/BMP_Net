The code with paper "BMP-Net: A Semantic-Segmentation Model for Road Extraction in Remote-Sensing Images".

With the continuous development of urban infrastructure, accu-rate extraction of road networks is crucial for urban planning and management. Semantic-segmentation techniques provide an effective solution for road extraction by assigning each pixel in an image to a corresponding semantic category. Traditional convolu-tional neural network (CNN)–based semantic-segmentation methods often involve insufficient capture of global context in-formation and uneven processing of information at different scales, which affect the segmentation accuracy. In this paper, we propose the novel network architecture BMP-Net, including Bi-Former, a multilayer feature cross-fusion module (MFCMF), and a pyramid information-fusion module (PIFM). BiFormer achieves more flexible arithmetic allocation through a bi-level routing at-tention, and it has higher computational efficiency when extract-ing features. In MFCMF, feature maps of different layers are cross-fertilized to capture richer semantic information. In PIFM, shifted MLP (multilayer perceptron) and partial large-kernel convolution are used for feature extraction to obtain a global sen-sory view while reducing the computational effort as much as possible. Through experiments on the Massachusetts and CHN6-CUG datasets, the effectiveness and performance enhancement of the model in road extraction tasks are demonstrated, providing new ideas for future road-extraction tasks. The code is available at https://github.com/ziyanpeng/BMP-Net.

The proposed network architecture of BMP-Net：
![image](https://github.com/ziyanpeng/BMP_Net/blob/master/network.png)

The structure of PIFM：
![image](https://github.com/ziyanpeng/BMP_Net/blob/master/PIFM.JPG)

Visual segmentation results：
![image](https://github.com/ziyanpeng/BMP_Net/blob/master/predictplot.png)

Our training data has been uploaded to Baidu Cloud: 链接：https://pan.baidu.com/s/1wfoy_7YiFUGPFvMHkc3e0Q 
提取码：1234 

If you have any questions during use, please contact us. Our email address is: m210200619@st.shou.edu.cn.
