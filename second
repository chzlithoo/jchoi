<p><div class="toc">
<ul>
<li><a href="#ch-6-feedforward-deep-networks">Ch 6. Feedforward Deep Networks</a><ul>
<li><a href="#61-vanilla-shallow-mlps">6.1 Vanilla (Shallow) MLPs</a><ul>
<li><ul>
<li><a href="#example-611-vanilla-shallow-multi-layer-neural-network-for-regression">Example 6.1.1. Vanilla (Shallow) Multi-Layer Neural Network for Regression</a></li>
</ul>
</li>
</ul>
</li>
<li><a href="#62-estimating-conditional-statistics">6.2 Estimating Conditional Statistics</a></li>
<li><a href="#63-parametrizing-a-learned-predictor">6.3 Parametrizing a Learned Predictor</a><ul>
<li><ul>
<li><a href="#631-family-of-functions">6.3.1 Family of Functions</a></li>
<li><a href="#632-loss-function-and-conditional-log-likelihood">6.3.2 Loss Function and Conditional Log-Likelihood</a></li>
<li><a href="#633-cost-functions-for-neural-networks">6.3.3 Cost Functions For Neural Networks</a></li>
<li><a href="#634-optimization-procedure">6.3.4 Optimization Procedure</a></li>
</ul>
</li>
</ul>
</li>
<li><a href="#64-flow-graphs-and-back-propagation">6.4 Flow Graphs and Back-Propagation</a><ul>
<li><ul>
<li><a href="#641-chain-rule">6.4.1 Chain Rule</a></li>
<li><a href="#642-back-propagation-in-an-mlp">6.4.2 Back-Propagation in an MLP</a></li>
<li><a href="#643-back-propagation-in-a-general-flow-graph">6.4.3 Back-Propagation in a General Flow Graph</a></li>
<li><a href="#644-symbolic-back-propagation-and-automatic-differentiation">6.4.4 Symbolic Back-propagation and Automatic Differentiation</a></li>
<li><a href="#645-back-propagation-through-random-operations-and-graphical-models">6.4.5 Back-propagation Through Random Operations and Graphical Models</a></li>
</ul>
</li>
</ul>
</li>
<li><a href="#65-universal-approximation-properties-and-depth">6.5 Universal Approximation Properties and Depth</a></li>
<li><a href="#66-feature-representation-learning">6.6 Feature / Representation Learning</a></li>
<li><a href="#67-piecewise-linear-hidden-units">6.7 Piecewise Linear Hidden Units</a></li>
</ul>
</li>
</ul>
</div>
</p>



<h1 id="ch-6-feedforward-deep-networks">Ch 6. Feedforward Deep Networks</h1>

<ul>
<li>a.k.a.  multilayer perceptrons (MLPs)</li>
<li>parametric functions deﬁned by composing together many parametric functions</li>
<li>multiple inputs and multiple outputs</li>
</ul>

<p><strong>terminology</strong></p>

<ul>
<li>a <strong><em>layer</em></strong> of the network: each sub-function</li>
<li>a <strong><em>unit</em></strong> (or a <strong><em>feature</em></strong>): each scalar output of one of these functions <br>
<img src="https://upload.wikimedia.org/wikipedia/en/5/54/Feed_forward_neural_net.gif" alt="enter image description here" title=""></li>
</ul>

<p><strong>width and depth of a machine learning model</strong></p>

<ul>
<li><strong><em>width</em></strong> : the number of units in each layer</li>
<li><p><strong><em>depth</em></strong> : the number of layer</p></li>
<li><p>conceptually simple example of an algorithm that captures the many advantages that come from having significant width and depth. </p></li>
<li><p>the key technology underlying most of the contemporary commercial applications of deep learning to large datasets.</p></li>
<li><p>traditional machine learning algorithms (linear regression, linear classifiers, logistic regression and kernel machines): the functions are non-linear in the space of inputs x, but they are linear in some other pre-defined space.</p></li>
<li><p>Neural networks allow us to learn new kinds of non-linearity:  <br>
to learn the features provided to a linear model</p></li>
</ul>

<hr>



<h2 id="61-vanilla-shallow-mlps">6.1 Vanilla (Shallow) MLPs</h2>

<ul>
<li>among the ﬁrst and most successful learning algorithms <br>
(Rumelhart et al., 1986e,c)</li>
<li>learn at least one function deﬁning the features, as well as a (typically linear) function mapping from features to output</li>
<li><p>hidden layers:  <br>
the layers of the network that correspond to features rather than outputs.</p></li>
<li><p>a <strong><em>vanilla MLP</em></strong>:  architecture with a <strong><em>single</em></strong> hidden layer <br>
<img src="https://lh3.googleusercontent.com/OF3UmkAbqhapN5gOQPpsJ2KtfCJDG3BWNkklTe-vDlw=s0" alt="enter image description here" title="Screen Shot 2015-08-31 at 9.18.49 PM.png"></p></li>
<li>hidden unit vector: <script type="math/tex" id="MathJax-Element-5">h=sigmoid(c+Wx)</script></li>
<li>output vector:  <script type="math/tex" id="MathJax-Element-6">\hat { y } =b+Vh</script></li>
</ul>



<hr>



<h4 id="example-611-vanilla-shallow-multi-layer-neural-network-for-regression">Example 6.1.1. Vanilla (Shallow) Multi-Layer Neural Network for Regression</h4>

<ul>
<li><p>the input-output functions : <br>
<script type="math/tex; mode=display" id="MathJax-Element-7">
{ f }_{ \theta  }(x)= b+Vsigmoid(c+Wx)\\ \\where \\ \\ sigmoid(a) = 1/(1+{ e }^{ -a })
\\
</script></p></li>
<li><p>input: <script type="math/tex" id="MathJax-Element-8">x\in { \Re  }^{ { n }_{ i } }</script></p></li>
<li><p>hidden layer output:  <br>
<script type="math/tex" id="MathJax-Element-9">h =sigmoid(c+Wx)</script></p></li>
<li><p>parameters: <script type="math/tex" id="MathJax-Element-10">\theta =(b,c,V,W)</script></p></li>
<li><p>weight matrices: <script type="math/tex" id="MathJax-Element-11">V\in { \Re  }^{ { n }_{ o }\times { n }_{ h } }</script> , <script type="math/tex" id="MathJax-Element-12">W\in { \Re  }^{ { n }_{ h }\times { n }_{ i } }</script></p></li>
<li><p>loss function: <script type="math/tex" id="MathJax-Element-13">L(\hat { y } -y)= { \left\| \hat { y } -y \right\|  }^{ 2 }</script></p></li>
<li><p><script type="math/tex" id="MathJax-Element-14">{ L }^{ 2 }</script> decay (regularizer): <script type="math/tex" id="MathJax-Element-15">{ \left\| \omega  \right\|  }^{ 2 }\quad =\quad \left( \sum _{ ij }{ { W }_{ ij }^{ 2 } } +\sum _{ ki }{{ W } _{ ki }^{ 2 }} \right) </script></p></li>
<li><p>minimize by: <br>
<script type="math/tex; mode=display" id="MathJax-Element-16">
 J(\theta )=\lambda { \left\| \omega  \right\|  }^{ 2 }+\frac { 1 }{ n } \sum _{ l=1 }^{ n }{ { \left\| { y }^{ (t) }-(b+Vsigmoid(c+W{ x }^{ (t) })) \right\|  }^{ 2 } } 
 </script></p></li>
<li><p>training procedure (stochastic gradient descent): <br>
<script type="math/tex; mode=display" id="MathJax-Element-17">
\omega \quad \leftarrow \quad \omega \quad -\epsilon \left( 2\lambda +{ \nabla  }_{ \omega  }L({ f }_{ \theta  }({ x }^{ (t) },{ y }^{ (t) }) \right) \\ \beta \quad \leftarrow \quad \beta \quad -\epsilon { \nabla  }_{ \beta  }L\left( { f }_{ \theta  }({ x }^{ (t) }),{ y }^{ (t) } \right) 
 </script></p>

<hr></li>
<li><p>MLPs can learn powerful non-linear transformations: in fact, with enough hidden units they can represent arbitrarily complex but smooth functions, they can be universal approximators (Section 6.5). </p></li>
<li><p>This is achieved by composing simple but non-linear learned transformations.</p></li>
<li><p>By transforming the data non-linearly into a new space, a classification problem that was not linearly separable (not solvable by a linear classifier) can become separable, as illustrated in Figures 6.2 and 6.3.</p></li>
</ul>

<p><img src="https://lh3.googleusercontent.com/FaWeDeUwADxj5bqPvb9EH-DyL3toquiUFIN27QLF4us=s0" alt="enter image description here" title="Screen Shot 2015-08-31 at 8.30.31 PM.png"></p>

<p><img src="https://lh3.googleusercontent.com/vXtj5jxocQ8zbml_u8QWYsALFIKxZS5NbFsFq1Sp_Ss=s0" alt="enter image description here" title="Screen Shot 2015-08-31 at 8.30.35 PM.png"></p>

<hr>



<h2 id="62-estimating-conditional-statistics">6.2 Estimating Conditional Statistics</h2>

<ul>
<li>linear regression : any function f by defining the mean squared error of f <br>
<script type="math/tex; mode=display" id="MathJax-Element-172">
E[{ ||y−f(x)|| }^{ 2 }]
</script></li>
<li><p>generalization: minimizing it yields an estimator of the conditional expectation of the output variable y given the input variable x <br>
<script type="math/tex; mode=display" id="MathJax-Element-173">
E_{ p(x,y) }\left[ { \left\| y-f(x) \right\|  }^{ 2 } \right] =E_{ p(x,y) }\left[ y|x \right] 
</script></p></li>
<li><p>generalize conditional maximum likelihood (introduced in Section 5.6.1) to other distributions than the Gaussian</p></li>
</ul>

<hr>

<h2 id="63-parametrizing-a-learned-predictor">6.3 Parametrizing a Learned Predictor</h2>



<h4 id="631-family-of-functions">6.3.1 Family of Functions</h4>

<ul>
<li><strong>motivation</strong>: to compose <em>simple transformations</em> in order to obtain  <br>
<em>highly non-linear</em> ones</li>
<li>(MLPs compose affine transformations and element-wise non-linearities)</li>
<li>hyperbolic tangent activation functions: <br>
<script type="math/tex; mode=display" id="MathJax-Element-897">
 { h }^{ k }=tanh({ b }^{ k }+{ W }^{ k }{ h }^{ k-1 })
</script></li>
<li>the input of the neural net: <script type="math/tex" id="MathJax-Element-898">{ h }^{ 0 }=x</script></li>
<li><p>theoutputofthe k-th hidden layer: <script type="math/tex" id="MathJax-Element-899">{ h }^{ k }</script></p></li>
<li><p>affine transformation <script type="math/tex" id="MathJax-Element-900">a = b+Wx</script> , elementwise <br>
<script type="math/tex; mode=display" id="MathJax-Element-901">
h=\phi (a)⇔{ h }_{ i }=\phi ({ a }_{ i })=\phi ({ b }_{ i }+{ W }_{ i,: }x)
</script></p></li>
<li><p>non-linear neural network activation functions:</p>

<h6 id="rectifier-or-rectified-linear-unit-relu-or-positive-part">Rectifier or rectified linear unit (ReLU) or positive part</h6>

<h6 id="hyperbolic-tangent">Hyperbolic tangent</h6>

<h6 id="sigmoid">Sigmoid</h6>

<h6 id="softmax">Softmax</h6>

<h6 id="radial-basis-function-or-rbf">Radial basis function or RBF</h6>

<h6 id="softplus">Softplus</h6>

<h6 id="hard-tanh">Hard tanh</h6>

<h6 id="absolute-value-rectification">Absolute value rectification</h6>

<h6 id="maxout">Maxout</h6></li>
<li><p>the structure (also called architecture) of the family of input-output functions can be varied in many ways:  <br>
<em>convolutional networks</em>,  <br>
<em>recurrent networks</em></p></li>
</ul>

<h4 id="632-loss-function-and-conditional-log-likelihood">6.3.2 Loss Function and Conditional Log-Likelihood</h4>

<ul>
<li><p>In the 80’s and 90’s the most commonly used loss function was the squared error <br>
<script type="math/tex; mode=display" id="MathJax-Element-711">
L({ f }_{ θ }(x),y)={ ||fθ(x)−y|| }^{ 2 }
</script></p></li>
<li><p>if f is unrestricted (non- parametric), <br>
<script type="math/tex; mode=display" id="MathJax-Element-712">
 f(x) = E[y | x = x]
</script></p></li>
<li><p>Replacing the squared error by an absolute value makes the neural network try to estimate not the conditional expectation but the conditional median</p></li>
<li><p><strong>cross entropy objective function</strong>: when y is a discrete label, i.e., for classification problems, other loss functions such as the Bernoulli negative log-likelihood4 have been found to be more appropriate than the squared error. (<script type="math/tex" id="MathJax-Element-713">y∈{ \left\{ 0,1 \right\}  }</script>)</p></li>
</ul>

<p><script type="math/tex; mode=display" id="MathJax-Element-906">
L({ f }_{ θ }(x),y)=−ylog{ f }_{ θ }(x)−(1−y)log(1−{ f }_{ θ }(x))
</script></p>

<ul>
<li><script type="math/tex" id="MathJax-Element-907">{f}_{\theta}(x)</script> to be strictly between 0 to 1: use the sigmoid as non-linearity for the output layer(matches well with the binomial negative log-likelihood cost function)</li>
</ul>

<h5 id="learning-a-conditional-probability-model">Learning a Conditional Probability Model</h5>

<ul>
<li>loss function as corresponding to a conditional log-likelihood, i.e., the negative log-likelihood (NLL) cost function <br>
<script type="math/tex; mode=display" id="MathJax-Element-990">
{ L }_{ NLL }({ f }_{ \theta  }(x),y)=−logP(y=y|x=x;θ)
</script></li>
<li>example) if y is a continuous random variable and we assume that, given x, it has a Gaussian distribution with mean <script type="math/tex" id="MathJax-Element-991">{f}_{θ}</script>(x) and variance <script type="math/tex" id="MathJax-Element-992">{\sigma}^{2}</script> <br>
<script type="math/tex; mode=display" id="MathJax-Element-993">
−logP(y|x;θ)=\frac { 1 }{ 2 } { ({ f }_{ \theta  }(x)−y) }^{ 1 }/{ σ }^{ 2 }+log(2π{ σ }^{ 2 })
</script></li>
<li><p>minimizing this negative log-likelihood is therefore equivalent to minimizing the squared error loss.</p></li>
<li><p>for discrete variables, the binomial negative log-likelihood cost func- tion corresponds to the conditional log-likelihood associated with the Bernoulli distribution (also known as cross entropy) with probability <script type="math/tex" id="MathJax-Element-994">p = {f}_{θ}(x)</script> of generating y = 1 given x =<script type="math/tex" id="MathJax-Element-995"> x</script> <br>
<script type="math/tex; mode=display" id="MathJax-Element-996">
{L}_{NLL}=−logP(y|x;θ)={−1}_{y=1}{logp−1}_{y=0}log(1−p)\\ =−ylog{f}_{θ}(x)−(1−y)log(1−{f}_{θ}(x))
</script></p></li>
</ul>

<h5 id="softmax-1">Softmax</h5>

<ul>
<li>designed for the purpose of specifying multinoulli distributions: <br>
<script type="math/tex; mode=display" id="MathJax-Element-1040">
p=softmax(a)\Longleftrightarrow { p }_{ i }=\frac { { e }^{ { a }_{ i } } }{ \sum { _{ j }^{  }{ { e }^{ { a }_{ j } } } }  } 
</script></li>
<li>consider the gradient with respect to the scores <script type="math/tex" id="MathJax-Element-1041">a</script>. <br>
<script type="math/tex; mode=display" id="MathJax-Element-1042">
\frac { ∂ }{ ∂{ a }_{ k } } { L }_{ NLL }(p,y)=\frac { ∂ }{ ∂{ a }_{ k } } (−log{ p }_{ y })=\frac { ∂ }{ ∂{ a }_{ k } } ({ −a }_{ y }+log\sum _{ j }^{  }{ { e }^{ { a }_{ j } } } )\\ ={ −1 }_{ y=k }+\frac { { e }^{ { a }_{ k } } }{ \sum _{ j }^{  }{ { e }^{ { a }_{ j } } }  } ={ p }_{ k }-{1}_{y=k}
</script> <br>
or <br>
<script type="math/tex; mode=display" id="MathJax-Element-1043">
\frac { ∂ }{ ∂{ a }_{ k } } { L }_{ NLL }(p,y)=(p-{e}_{y})
</script></li>
</ul>

<h4 id="633-cost-functions-for-neural-networks">6.3.3 Cost Functions For Neural Networks</h4>

<ul>
<li>a good choice for the criterion is maximum likelihood regularized with dropout, possibly also with weight decay.</li>
</ul>

<h4 id="634-optimization-procedure">6.3.4 Optimization Procedure</h4>

<ul>
<li>a good choice for the optimization algorithm for a feedforward network is usually stochastic gradient descent with momentum.</li>
</ul>

<h2 id="64-flow-graphs-and-back-propagation">6.4 Flow Graphs and Back-Propagation</h2>

<p><strong>back-propagation</strong> <br>
- it just means the method for computing gradients in such networks <br>
- the output of the function to differentiate (e.g., the training criterion J) is a scalar and we are interested in its derivative with respect to a set of parameters (considered to be the elements of a vector θ), or equivalently, a set of inputs <br>
- The partial derivative of J with respect to θ (called the gradient) tells us whether θ should be increased or de- creased in order to decrease J</p>

<ul>
<li>the partial derivative of the cost J with respect to parameters θ can be <em>decomposed recursively</em> by taking into consideration the composition of functions that relate θ to J , via intermediate quantities that mediate that influence, e.g., the activations of hidden units in a deep neural network.</li>
</ul>

<h4 id="641-chain-rule">6.4.1 Chain Rule</h4>

<ul>
<li>the locally linear influence of a variable x on another one y</li>
<li>output: the cost, or objective function, <script type="math/tex" id="MathJax-Element-159">z = J(g(θ))</script></li>
</ul>

<h4 id="642-back-propagation-in-an-mlp">6.4.2 Back-Propagation in an MLP</h4>



<h4 id="643-back-propagation-in-a-general-flow-graph">6.4.3 Back-Propagation in a General Flow Graph</h4>



<h4 id="644-symbolic-back-propagation-and-automatic-differentiation">6.4.4 Symbolic Back-propagation and Automatic Differentiation</h4>



<h4 id="645-back-propagation-through-random-operations-and-graphical-models">6.4.5 Back-propagation Through Random Operations and Graphical Models</h4>



<h2 id="65-universal-approximation-properties-and-depth">6.5 Universal Approximation Properties and Depth</h2>

<ul>
<li>feedforward networks with hidden layers provide a universal approximation framework.</li>
<li>universal approximation theorem (Hornik et al., 1989; Cybenko, 1989) states that a feedforward network with a linear output layer and at least one hidden layer with any “squashing” activation function (such as the logistic sigmoid activation function) can approximate any Borel measurable function from one finite-dimensional space to another with any desired non-zero amount of error, provided that the network is given enough hidden units.</li>
</ul>

<h2 id="66-feature-representation-learning">6.6 Feature / Representation Learning</h2>

<ul>
<li><p>limitations of convex optimization problem on the representational capacity: many tasks, for a given choice of input representation x (the raw input features), cannot be solved by using only a linear predictor.</p></li>
<li><p>solutions: kernel machine, manually engineer the representation or features φ(x),  or learn the features.</p></li>
</ul>

<h2 id="67-piecewise-linear-hidden-units">6.7 Piecewise Linear Hidden Units</h2>

<ul>
<li></li>
<li><blockquote>
  <p>Written with <a href="https://stackedit.io/">StackEdit</a>.</p>
</blockquote></li>
</ul>