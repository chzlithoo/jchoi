<p><div class="toc">
<ul>
<li><a href="#ch-6-feedforward-deep-networks">Ch 6. Feedforward Deep Networks</a><ul>
<li><ul>
<li><ul>
<li><ul>
<li><a href="#terminology">terminology</a></li>
<li><a href="#width-and-depth-of-a-machine-learning-model">width and depth of a machine learning model</a></li>
</ul>
</li>
</ul>
</li>
</ul>
</li>
<li><a href="#61-vanilla-shallow-mlps">6.1 Vanilla (Shallow) MLPs</a></li>
<li><a href="#62-estimating-conditional-statistics">6.2 Estimating Conditional Statistics</a></li>
<li><a href="#63-parametrizing-a-learned-predictor">6.3 Parametrizing a Learned Predictor</a></li>
<li><a href="#64-flow-graphs-and-back-propagation">6.4 Flow Graphs and Back-Propagation</a></li>
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



<h5 id="terminology">terminology</h5>

<ul>
<li>a <strong><em>layer</em></strong> of the network: each sub-function</li>
<li>a <strong><em>unit</em></strong> (or a <strong><em>feature</em></strong>): each scalar output of one of these functions <br>
<img src="https://upload.wikimedia.org/wikipedia/en/5/54/Feed_forward_neural_net.gif" alt="enter image description here" title=""></li>
</ul>

<h5 id="width-and-depth-of-a-machine-learning-model">width and depth of a machine learning model</h5>

<ul>
<li><strong><em>width</em></strong> : the number of units in each layer</li>
<li><strong><em>depth</em></strong> : the number of layer</li>
</ul>

<p>Feedforward deep networks provide a conceptually simple example of an algorithm that captures the many advantages that come from having significant width and depth. Feedforward deep networks are also the key technology underlying most of the contemporary commercial applications of deep learning to large datasets.</p>

<ul>
<li>traditional machine learning algorithms, including linear regression, linear classifiers, logistic regression and kernel machines. All of these algorithms work by applying a linear transformation to a fixed set of features. These algorithms can learn non-linear functions, but the non-linear part is fixed. In other words, the functions are non-linear in the space of inputs x, but they are linear in some other pre-defined space.</li>
<li>Neural networks allow us to learn new kinds of non-linearity. Another way to view this idea is that neural networks allow us to learn the features provided to a linear model. From this point of view, neural networks allow us to automate the design of features—a task that until recently was performed gradually and collectively, by the combined efforts of an entire community of researchers</li>
</ul>

<hr>



<h2 id="61-vanilla-shallow-mlps">6.1 Vanilla (Shallow) MLPs</h2>

<ul>
<li>among the ﬁrst and most successful learning algorithms <br>
(Rumelhart et al., 1986e,c)</li>
<li>learn at least one function deﬁning the features, as well as a (typically linear) function mapping from features to output</li>
<li><p>hidden layers:  <br>
the layers of the network that correspond to features rather than outputs.</p></li>
<li><p>a <strong><em>vanilla MLP</em></strong> architecture with a single hidden layer :</p></li>
</ul>



<h4 id="example-611-vanilla-shallow-multi-layer-neural-network-for-regression">Example 6.1.1. Vanilla (Shallow) Multi-Layer Neural Network for Regression</h4>

<ul>
<li><p>the input-output functions : <br>
<script type="math/tex; mode=display" id="MathJax-Element-578">
{ f }_{ \theta  }(x)= b+Vsigmoid(c+Wx)\\ where\\ sigmoid(a) = 1/(1+{ e }^{ -a })
\\
</script></p></li>
<li><p>input: <script type="math/tex" id="MathJax-Element-579">x\in { \Re  }^{ { n }_{ i } }</script></p></li>
<li>hidden layer output: <script type="math/tex" id="MathJax-Element-580">h =sigmoid(c+Wx)</script></li>
<li>parameters: <script type="math/tex" id="MathJax-Element-581">\theta =(b,c,V,W)</script></li>
<li>weight matrices: <script type="math/tex" id="MathJax-Element-582">V\in { \Re  }^{ { n }_{ o }\times { n }_{ h } }</script> , <script type="math/tex" id="MathJax-Element-583">W\in { \Re  }^{ { n }_{ h }\times { n }_{ i } }</script></li>
<li>loss function: <script type="math/tex" id="MathJax-Element-584">L(\hat { y } -y)= { \left\| \hat { y } -y \right\|  }^{ 2 }</script></li>
<li><p>minimize by: <br>
<script type="math/tex; mode=display" id="MathJax-Element-585">
 J(\theta )=\lambda { \left\| \omega  \right\|  }^{ 2 }+\frac { 1 }{ n } \sum _{ l=1 }^{ n }{ { \left\| { y }^{ (t) }-(b+Vsigmoid(c+W{ x }^{ (t) })) \right\|  }^{ 2 } } 
 </script></p></li>
<li><p>training procedure (stochastic gradient descent): <br>
<script type="math/tex; mode=display" id="MathJax-Element-586">
\omega \quad \leftarrow \quad \omega \quad -\epsilon \left( 2\lambda +{ \nabla  }_{ \omega  }L({ f }_{ \theta  }({ x }^{ (t) },{ y }^{ (t) }) \right) \\ \beta \quad \leftarrow \quad \beta \quad -\epsilon { \nabla  }_{ \beta  }L\left( { f }_{ \theta  }({ x }^{ (t) }),{ y }^{ (t) } \right) 
 </script></p></li>
<li><p>MLPs can learn powerful non-linear transformations: in fact, with enough hidden units they can represent arbitrarily complex but smooth functions, they can be universal approximators(Section 6.5). </p></li>
<li>This is achieved by composing simple but non-linear learned transformations.</li>
<li>By transforming the data non-linearly into a new space, a classification problem that was not linearly separable (not solvable by a linear classifier) can become separable, as illustrated in Figures 6.2 and 6.3.</li>
</ul>

<h2 id="62-estimating-conditional-statistics">6.2 Estimating Conditional Statistics</h2>

<p><script type="math/tex; mode=display" id="MathJax-Element-126">
E_{ p(x,y) }\left[ { \left\| y-f(x) \right\|  }^{ 2 } \right] =E_{ p(x,y) }\left[ y|x \right] 
</script> <br>
- to be updated</p>

<h2 id="63-parametrizing-a-learned-predictor">6.3 Parametrizing a Learned Predictor</h2>

<ul>
<li>to be updated</li>
</ul>

<h2 id="64-flow-graphs-and-back-propagation">6.4 Flow Graphs and Back-Propagation</h2>

<ul>
<li>to be updated</li>
</ul>

<h2 id="65-universal-approximation-properties-and-depth">6.5 Universal Approximation Properties and Depth</h2>

<ul>
<li>to be updated</li>
</ul>

<h2 id="66-feature-representation-learning">6.6 Feature / Representation Learning</h2>

<ul>
<li>to be updated</li>
</ul>

<h2 id="67-piecewise-linear-hidden-units">6.7 Piecewise Linear Hidden Units</h2>

<ul>
<li><p>to be updated</p></li>
<li><blockquote>
  <p>Written with <a href="https://stackedit.io/">StackEdit</a>.</p>
</blockquote></li>
</ul>