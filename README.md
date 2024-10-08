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

<h3>Accuracy Comparison Table</h3>

<p>I conducted an experiment where I performed inference on the MMLU (Massive Multitask Language Understanding) dataset using various models. The goal was to evaluate the accuracy of each model across a wide range of subjects. After obtaining the inference results, I applied a natural language processing (NLP) algorithm to compare the AI-generated answers with the expected correct answers from the dataset.</p>

<p>The comparison was done using a fuzzy string matching algorithm, which assesses the similarity between the AI's answer and the correct answer.</p>

<p>This function works by first preprocessing the answers to ensure consistency in comparison. It then calculates a similarity ratio using the fuzzy string matching technique. The function checks whether the AI's answer contains the correct answer, or vice versa, and whether the similarity ratio exceeds a threshold of 80%. Additionally, it verifies if the correct answer index is present in the AI's response.</p>

<p>By applying this algorithm, I was able to accurately determine the correctness of the AI-generated answers, allowing for a more nuanced evaluation of the model's performance on the MMLU dataset.</p>

<table>
  <thead>
    <tr>
      <th>Subject/model</th>
      <th>Minitron</th>
      <th>Mistral</th>
      <th>Llama</th>
      <th>Gemma</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>abstract_algebra</td>
      <td>0.364</td>
      <td>0.545</td>
      <td>0.455</td>
      <td>0.273</td>
    </tr>
    <tr>
      <td>anatomy</td>
      <td>0.071</td>
      <td>0.357</td>
      <td>0.143</td>
      <td>0.571</td>
    </tr>
    <tr>
      <td>astronomy</td>
      <td>0.063</td>
      <td>0.313</td>
      <td>0.063</td>
      <td>0.375</td>
    </tr>
    <tr>
      <td>business_ethics</td>
      <td>0.091</td>
      <td>0.727</td>
      <td>0.091</td>
      <td>0.273</td>
    </tr>
    <tr>
      <td>clinical_knowledge</td>
      <td>0.069</td>
      <td>0.414</td>
      <td>0.241</td>
      <td>0.345</td>
    </tr>
    <tr>
      <td>college_biology</td>
      <td>0.063</td>
      <td>0.375</td>
      <td>0.250</td>
      <td>0.500</td>
    </tr>
    <tr>
      <td>college_chemistry</td>
      <td>0.250</td>
      <td>0.250</td>
      <td>0.250</td>
      <td>0.625</td>
    </tr>
    <tr>
      <td>college_computer_science</td>
      <td>0.091</td>
      <td>0.273</td>
      <td>0.182</td>
      <td>0.273</td>
    </tr>
    <tr>
      <td>college_mathematics</td>
      <td>0.091</td>
      <td>0.273</td>
      <td>0.273</td>
      <td>0.182</td>
    </tr>
    <tr>
      <td>college_medicine</td>
      <td>0.136</td>
      <td>0.409</td>
      <td>0.273</td>
      <td>0.409</td>
    </tr>
    <tr>
      <td>college_physics</td>
      <td>0.182</td>
      <td>0.273</td>
      <td>0.364</td>
      <td>0.273</td>
    </tr>
    <tr>
      <td>computer_security</td>
      <td>0.091</td>
      <td>0.455</td>
      <td>0.273</td>
      <td>0.364</td>
    </tr>
    <tr>
      <td>conceptual_physics</td>
      <td>0.231</td>
      <td>0.346</td>
      <td>0.385</td>
      <td>0.423</td>
    </tr>
    <tr>
      <td>econometrics</td>
      <td>0.083</td>
      <td>0.667</td>
      <td>0.333</td>
      <td>0.333</td>
    </tr>
    <tr>
      <td>electrical_engineering</td>
      <td>0.188</td>
      <td>0.375</td>
      <td>0.313</td>
      <td>0.375</td>
    </tr>
    <tr>
      <td>elementary_mathematics</td>
      <td>0.195</td>
      <td>0.293</td>
      <td>0.317</td>
      <td>0.341</td>
    </tr>
    <tr>
      <td>formal_logic</td>
      <td>0.143</td>
      <td>0.214</td>
      <td>0.286</td>
      <td>0.214</td>
    </tr>
    <tr>
      <td>global_facts</td>
      <td>0.100</td>
      <td>0.200</td>
      <td>0.500</td>
      <td>0.200</td>
    </tr>
    <tr>
      <td>high_school_biology</td>
      <td>0.125</td>
      <td>0.438</td>
      <td>0.156</td>
      <td>0.313</td>
    </tr>
    <tr>
      <td>high_school_chemistry</td>
      <td>0.091</td>
      <td>0.182</td>
      <td>0.091</td>
      <td>0.182</td>
    </tr>
    <tr>
      <td>high_school_computer_science</td>
      <td>0.111</td>
      <td>0.778</td>
      <td>0.111</td>
      <td>0.111</td>
    </tr>
    <tr>
      <td>high_school_european_history</td>
      <td>0.056</td>
      <td>0.556</td>
      <td>0.111</td>
      <td>0.556</td>
    </tr>
    <tr>
      <td>high_school_geography</td>
      <td>0.136</td>
      <td>0.455</td>
      <td>0.409</td>
      <td>0.545</td>
    </tr>
    <tr>
      <td>high_school_government_and_politics</td>
      <td>0.238</td>
      <td>0.476</td>
      <td>0.381</td>
      <td>0.476</td>
    </tr>
    <tr>
      <td>high_school_macroeconomics</td>
      <td>0.070</td>
      <td>0.512</td>
      <td>0.186</td>
      <td>0.349</td>
    </tr>
    <tr>
      <td>high_school_mathematics</td>
      <td>0.069</td>
      <td>0.448</td>
      <td>0.414</td>
      <td>0.241</td>
    </tr>
    <tr>
      <td>high_school_microeconomics</td>
      <td>0.115</td>
      <td>0.385</td>
      <td>0.115</td>
      <td>0.269</td>
    </tr>
    <tr>
      <td>high_school_physics</td>
      <td>0.000</td>
      <td>0.235</td>
      <td>0.000</td>
      <td>0.118</td>
    </tr>
    <tr>
      <td>high_school_psychology</td>
      <td>0.200</td>
      <td>0.617</td>
      <td>0.367</td>
      <td>0.483</td>
    </tr>
    <tr>
      <td>high_school_statistics</td>
      <td>0.043</td>
      <td>0.435</td>
      <td>0.261</td>
      <td>0.391</td>
    </tr>
    <tr>
      <td>high_school_us_history</td>
      <td>0.045</td>
      <td>0.545</td>
      <td>0.136</td>
      <td>0.409</td>
    </tr>
    <tr>
      <td>high_school_world_history</td>
      <td>0.038</td>
      <td>0.462</td>
      <td>0.192</td>
      <td>0.231</td>
    </tr>
    <tr>
      <td>human_aging</td>
      <td>0.087</td>
      <td>0.348</td>
      <td>0.391</td>
      <td>0.348</td>
    </tr>
    <tr>
      <td>human_sexuality</td>
      <td>0.083</td>
      <td>0.250</td>
      <td>0.250</td>
      <td>0.250</td>
    </tr>
    <tr>
      <td>international_law</td>
      <td>0.077</td>
      <td>0.462</td>
      <td>0.077</td>
      <td>0.231</td>
    </tr>
    <tr>
      <td>jurisprudence</td>
      <td>0.091</td>
      <td>0.364</td>
      <td>0.091</td>
      <td>0.182</td>
    </tr>
    <tr>
      <td>logical_fallacies</td>
      <td>0.056</td>
      <td>0.500</td>
      <td>0.278</td>
      <td>0.278</td>
    </tr>
    <tr>
      <td>machine_learning</td>
      <td>0.364</td>
      <td>0.545</td>
      <td>0.273</td>
      <td>0.455</td>
    </tr>
    <tr>
      <td>management</td>
      <td>0.182</td>
      <td>0.455</td>
      <td>0.455</td>
      <td>0.636</td>
    </tr>
    <tr>
      <td>marketing</td>
      <td>0.280</td>
      <td>0.560</td>
      <td>0.440</td>
      <td>0.640</td>
    </tr>
    <tr>
      <td>medical_genetics</td>
      <td>0.091</td>
      <td>0.636</td>
      <td>0.182</td>
      <td>0.455</td>
    </tr>
    <tr>
      <td>miscellaneous</td>
      <td>0.430</td>
      <td>0.628</td>
      <td>0.593</td>
      <td>0.500</td>
    </tr>
    <tr>
      <td>moral_disputes</td>
      <td>0.158</td>
      <td>0.368</td>
      <td>0.132</td>
      <td>0.237</td>
    </tr>
    <tr>
      <td>moral_scenarios</td>
      <td>0.130</td>
      <td>0.380</td>
      <td>0.090</td>
      <td>0.360</td>
    </tr>
    <tr>
      <td>nutrition</td>
      <td>0.152</td>
      <td>0.273</td>
      <td>0.242</td>
      <td>0.394</td>
    </tr>
    <tr>
      <td>philosophy</td>
      <td>0.088</td>
      <td>0.382</td>
      <td>0.206</td>
      <td>0.471</td>
    </tr>
    <tr>
      <td>prehistory</td>
      <td>0.029</td>
      <td>0.343</td>
      <td>0.114</td>
      <td>0.343</td>
    </tr>
    <tr>
      <td>professional_accounting</td>
      <td>0.065</td>
      <td>0.258</td>
      <td>0.065</td>
      <td>0.194</td>
    </tr>
    <tr>
      <td>professional_law</td>
      <td>0.012</td>
      <td>0.165</td>
      <td>0.076</td>
      <td>0.271</td>
    </tr>
    <tr>
      <td>professional_medicine</td>
      <td>0.065</td>
      <td>0.419</td>
      <td>0.161</td>
      <td>0.290</td>
    </tr>
    <tr>
      <td>professional_psychology</td>
      <td>0.087</td>
      <td>0.493</td>
      <td>0.116</td>
      <td>0.333</td>
    </tr>
    <tr>
      <td>public_relations</td>
      <td>0.083</td>
      <td>0.500</td>
      <td>0.167</td>
      <td>0.250</td>
    </tr>
    <tr>
      <td>security_studies</td>
      <td>0.037</td>
      <td>0.222</td>
      <td>0.037</td>
      <td>0.296</td>
    </tr>
    <tr>
      <td>sociology</td>
      <td>0.045</td>
      <td>0.409</td>
      <td>0.045</td>
      <td>0.409</td>
    </tr>
    <tr>
      <td>us_foreign_policy</td>
      <td>0.091</td>
      <td>0.455</td>
      <td>0.000</td>
      <td>0.636</td>
    </tr>
    <tr>
      <td>virology</td>
      <td>0.222</td>
      <td>0.389</td>
      <td>0.278</td>
      <td>0.167</td>
    </tr>
    <tr>
      <td>world_religions</td>
      <td>0.316</td>
      <td>0.632</td>
      <td>0.579</td>
      <td>0.579</td>
    </tr>
    <tr>
      <td><strong>Moyenne</strong></td>
      <td><strong>12.558</strong></td>
      <td><strong>41.604</strong></td>
      <td><strong>23.258</strong></td>
      <td><strong>35.484</strong></td>
    </tr>
  </tbody>
</table>

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

