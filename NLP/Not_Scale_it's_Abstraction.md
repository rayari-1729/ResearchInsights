## The nature of Abstraction: reusing programs and representations 

* Abstraction is the key of Generalization.
* Hypothesis : The complexity & variability of any domain is the result of the repetition, composition, transformation, indtantiation of a small number
  of "Kernels od structure"
* Abstraction : Past experiences for resusable kernels of strucutre. 
* Intelligence : High sensitivity to similarities and isomorohisms, ability to recast old patterns into a new context. (relate with universe) 
</br>

**Abstraction**  
It is a spectrum from poitwise factoids to organize knowledge that works in many situations to generalizable models that work on any situation in a domain to the ability 
to produce new models to adapt to a new problem to the ability to producwe new models efficiently. 

**Central Question: Are LLM capable of abstraction? Can they reason?** 

</br>

### The two poles of abstraction : type 1 vs type 2

| Prototype-centric(value-centric) | Program-centric |
|----------------------------------|-----------------|
|Set of prototype + distance funciton. <br> Example: Classify face vs non-face using abstract features </br> | Graph of operators where input nodes can take different values within a type. <br>Example: function that sort a list </br>| 
|Continuous domain | discreate domain |
|Transformers are a great type 1 abstraction machine|How we get Transformer in type 2 ? ([Link](https://youtu.be/s7_NlkBwdj8?t=2163)) |

</br>

#### Scenario  
When we are playing chess, we are using type 2 when we calculate step by step. We are not calculating each moves, it;s cpmbinatorial explosion. We're handful of different options. So we use our intution which build up by lots  of games in  order ti narrow down the sort of discrete search that you perform when, you're calculating  so merginf type 1 to type 2  and that's why we can actually play chess using very very small cognitive resources compared to what a computer can do.


**Central question: Can we comvine both (type 1 & 2) into a super-approach?**

#### Merging Deep Learning and program synthesis 
* Using Deep Learning components side by sude with algorithmic components.
* Usig DL to guide program search.
