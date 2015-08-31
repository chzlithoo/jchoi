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
<li><p>a <strong><em>vanilla MLP</em></strong>:  architecture with a <strong><em>single</em></strong> hidden layer</p></li>
</ul>



<hr>



<h4 id="example-611-vanilla-shallow-multi-layer-neural-network-for-regression">Example 6.1.1. Vanilla (Shallow) Multi-Layer Neural Network for Regression</h4>

<ul>
<li><p>the input-output functions : <br>
<script type="math/tex; mode=display" id="MathJax-Element-1">
{ f }_{ \theta  }(x)= b+Vsigmoid(c+Wx)\\ \\where \\ \\ sigmoid(a) = 1/(1+{ e }^{ -a })
\\
</script></p></li>
<li><p>input: <script type="math/tex" id="MathJax-Element-2">x\in { \Re  }^{ { n }_{ i } }</script></p></li>
<li><p>hidden layer output:  <br>
-<script type="math/tex" id="MathJax-Element-3">h =sigmoid(c+Wx)</script></p></li>
<li><p>parameters: <script type="math/tex" id="MathJax-Element-4">\theta =(b,c,V,W)</script></p></li>
<li><p>weight matrices: <script type="math/tex" id="MathJax-Element-5">V\in { \Re  }^{ { n }_{ o }\times { n }_{ h } }</script> , <script type="math/tex" id="MathJax-Element-6">W\in { \Re  }^{ { n }_{ h }\times { n }_{ i } }</script></p></li>
<li><p>loss function: <script type="math/tex" id="MathJax-Element-7">L(\hat { y } -y)= { \left\| \hat { y } -y \right\|  }^{ 2 }</script></p></li>
<li><p>minimize by: <br>
<script type="math/tex; mode=display" id="MathJax-Element-8">
 J(\theta )=\lambda { \left\| \omega  \right\|  }^{ 2 }+\frac { 1 }{ n } \sum _{ l=1 }^{ n }{ { \left\| { y }^{ (t) }-(b+Vsigmoid(c+W{ x }^{ (t) })) \right\|  }^{ 2 } } 
 </script></p></li>
<li><p>training procedure (stochastic gradient descent): <br>
<script type="math/tex; mode=display" id="MathJax-Element-9">
\omega \quad \leftarrow \quad \omega \quad -\epsilon \left( 2\lambda +{ \nabla  }_{ \omega  }L({ f }_{ \theta  }({ x }^{ (t) },{ y }^{ (t) }) \right) \\ \beta \quad \leftarrow \quad \beta \quad -\epsilon { \nabla  }_{ \beta  }L\left( { f }_{ \theta  }({ x }^{ (t) }),{ y }^{ (t) } \right) 
 </script></p>

<hr></li>
<li><p>MLPs can learn powerful non-linear transformations: in fact, with enough hidden units they can represent arbitrarily complex but smooth functions, they can be universal approximators (Section 6.5). </p></li>
<li><p>This is achieved by composing simple but non-linear learned transformations.</p></li>
<li><p>By transforming the data non-linearly into a new space, a classification problem that was not linearly separable (not solvable by a linear classifier) can become separable, as illustrated in Figures 6.2 and 6.3.</p></li>
</ul>

<p><img src="https://drive.google.com/open?id=0BzCwlCXJz7_WZ3MxaVNUWDdUTWM" alt="6.2" title=""></p>

<p><img src="https://drive.google.com/open?id=0BzCwlCXJz7_WZ3MxaVNUWDdUTWM" alt="6.3" title=""></p>

<hr>



<h2 id="62-estimating-conditional-statistics">6.2 Estimating Conditional Statistics</h2>

<ul>
<li><p>generalization: minimizing it yields an estimator of the conditional expectation of the output variable y given the input variable x <br>
<script type="math/tex; mode=display" id="MathJax-Element-10">
E_{ p(x,y) }\left[ { \left\| y-f(x) \right\|  }^{ 2 } \right] =E_{ p(x,y) }\left[ y|x \right] 
</script></p></li>
<li><p>generalize conditional maximum likelihood (introduced in Section 5.6.1) to other distributions than the Gaussian</p></li>
</ul>

<hr>



<h2 id="63-parametrizing-a-learned-predictor">6.3 Parametrizing a Learned Predictor</h2>



<h4 id="631-family-of-functions">6.3.1 Family of Functions</h4>

<ul>
<li>to be updated</li>
</ul>



<h4 id="632-loss-function-and-conditional-log-likelihood">6.3.2 Loss Function and Conditional Log-Likelihood</h4>

<ul>
<li>to be updated</li>
</ul>



<h4 id="633-cost-functions-for-neural-networks">6.3.3 Cost Functions For Neural Networks</h4>



<h4 id="634-optimization-procedure">6.3.4 Optimization Procedure</h4>



<h2 id="64-flow-graphs-and-back-propagation">6.4 Flow Graphs and Back-Propagation</h2>

<ul>
<li>to be updated</li>
</ul>



<h4 id="641-chain-rule">6.4.1 Chain Rule</h4>



<h4 id="642-back-propagation-in-an-mlp">6.4.2 Back-Propagation in an MLP</h4>



<h4 id="643-back-propagation-in-a-general-flow-graph">6.4.3 Back-Propagation in a General Flow Graph</h4>



<h4 id="644-symbolic-back-propagation-and-automatic-differentiation">6.4.4 Symbolic Back-propagation and Automatic Differentiation</h4>



<h4 id="645-back-propagation-through-random-operations-and-graphical-models">6.4.5 Back-propagation Through Random Operations and Graphical Models</h4>



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