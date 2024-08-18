<h1>GPU Testing on <a href="https://app.primeintellect.ai/dashboard/create-cluster">PrimeIntellect</a></h1>
<img src="/images/prime-intellect.png" alt="PrimeIntellect interface">

<p>Welcome to the PrimeIntellect GPU testing documentation! PrimeIntellect is an innovative aggregator platform that simplifies the use of virtual machines (VMs) equipped with PyTorch images, enabling users to efficiently set up and execute deep learning models across various GPU configurations.</p>

<h2>Understanding the Test Framework</h2>
<img src="/images/excel.png" alt="GPU Test Result">
<p>The attached image provides a detailed overview of our testing metrics across different GPU models. For each GPU configuration, you can observe key parameters such as CPU, memory specifications, disk size, VRAM, and performance metrics such as time, RAM usage, accuracy, and cost associated with running specific deep learning models like Nvidia Minitron 8B, Mistral v0.3 (7B), Llama 3.1 (8B), and Gemma 2 (9B). This data is crucial for evaluating the efficiency and cost-effectiveness of deploying models on different GPU setups provided by PrimeIntellect.</p>

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

