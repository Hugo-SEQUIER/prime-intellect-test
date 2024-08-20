<h1>GPU Testing on <a href="https://app.primeintellect.ai/dashboard/create-cluster">PrimeIntellect</a></h1>
<img src="/images/prime-intellect.png" alt="PrimeIntellect interface">

<p>Welcome to the PrimeIntellect GPU testing documentation! PrimeIntellect is an innovative aggregator platform that simplifies the use of virtual machines (VMs) equipped with PyTorch images, enabling users to efficiently set up and execute deep learning models across various GPU configurations.</p>

<h2>Model Optimization Techniques: Pruning and Distillation</h2>
<h3>Pruning</h3>
<p>Pruning aims to make models smaller and more efficient by reducing the number of operational elements within them. The key methods include:</p>
<ul>
  <li><strong>Dropping Layers:</strong> Each layer in a neural network processes data differently, contributing to its ability to learn distinct aspects of the data. By selectively removing layers, the model can become leaner without significantly impacting performance.</li>
  <li><strong>Attention Heads:</strong> These are components of transformer models that help in parallel processing and understanding the context better rather than translating on a word-by-word basis. Pruning attention heads can streamline the model while maintaining its contextual awareness.</li>
  <li><strong>Embedding Channels:</strong> These are vector representations of data. Reducing the number of embedding channels can decrease the model's size and simplify the inputs it needs to process.</li>
</ul>

<h3>Distillation</h3>
<p>Distillation is a technique used to transfer knowledge from a large language model (LLM) to a smaller language model (SLM). This method involves training the smaller model to mimic the behavior and predictions of the larger one, effectively condensing the knowledge without needing the same computational resources.</p>

<h2>Understanding the Test Framework</h2>
<img src="/images/excel.png" alt="GPU Test Result">
<p>The attached image provides a detailed overview of our testing metrics across different GPU models. For each GPU configuration, you can observe key parameters such as CPU, memory specifications, disk size, VRAM, and performance metrics such as time, RAM usage, accuracy, and cost associated with running specific deep learning models like Nvidia Minitron 8B, Mistral v0.3 (7B), Llama 3.1 (8B), and Gemma 2 (9B). This data is crucial for evaluating the efficiency and cost-effectiveness of deploying models on different GPU setups provided by PrimeIntellect.</p>

<h3>Disclaimer on Accuracy Metrics</h3>
<p>Please note that the accuracy results mentioned here are based on a MMLU test conducted by NVIDIA, as documented in their study <a href="https://arxiv.org/abs/2407.14679">available here</a>. These results are used for reference; I am currently exploring methods to conduct my own effective testing to directly assess the training outcomes.</p>

<h2>Connecting to the VM</h2>
<img src="/images/connect_vm.png" alt="How to Connect to the VM">
<p>After launching an instance on PrimeIntellect, follow these steps to connect to your VM:</p>
<ol>
  <li><strong>Download the Private Key:</strong> Once your VM is ready, download the private key provided by PrimeIntellect. This key is necessary to securely connect to your VM.</li>
  <li><strong>Change Permissions on the Private Key:</strong> Before using the key, you must change its permissions to ensure that it is secure. Open a terminal on your computer and navigate to the directory where you downloaded the key. Then, execute the following command:
  <pre>chmod 400 [your-key-name].pem</pre></li>
  <li><strong>Connect to the VM:</strong> With the key's permissions set, you're ready to connect to the VM. In the same terminal window, use the connection command provided by PrimeIntellect. It will look something like this:
  <pre>ssh -i [your-key-name].pem ubuntu@[vm-ip-address]</pre></li>
</ol>
<p>Replace [your-key-name] with the name of your key file and [vm-ip-address] with the IP address provided for your VM.</p>
<h2>Setting Up Your Test Environment</h2>
<p>Once you've connected to a VM, setting up and running the test scripts is straightforward. Follow these steps:</p>

<ol>
  <li><strong>Clone the Repository</strong>
    <ul>
      <li>Open your VM's terminal.</li>
      <li>Execute the following command to clone the repository containing the test scripts:
        <pre>git clone https://github.com/Hugo-SEQUIER/prime-intellect-test.git</pre>
      </li>
    </ul>
  </li>
  <li><strong>Prepare the Scripts</strong>
    <ul>
      <li>Ensure that all scripts in the cloned repository are executable by running:
        <pre>find prime-intellect-test -type f -name "*.sh" -exec chmod +x {} \;</pre>
      </li>
    </ul>
  </li>
</ol>

<p>Now all you have to do is navigate the project and run the training script.</p>
<pre>cd prime-intellect-test
cd llama_8b
./training.sh
</pre>

<h2>Conclusion</h2>
<p>By following this guide, you can leverage PrimeIntellect's VMs to perform comprehensive benchmarks on different GPUs using the pre-configured PyTorch images. This will aid in making informed decisions about which GPU configuration best suits your deep learning tasks in terms of performance and cost efficiency.</p>

