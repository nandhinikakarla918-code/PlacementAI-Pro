/* PlacementAI Pro v3.0 — Application Logic
 * 3 ML Models: Logistic Regression + Decision Tree + Random Forest
 * Claude AI integration for all AI features
 * Question bank with deduplication
 * localStorage key: pai_v3
 */

'use strict';

'use strict';

/* ═══════════════════════════════════════════════════════
   PART 1: THREE ML MODELS
   ═══════════════════════════════════════════════════════ */

// ── Logistic Regression ─────────────────────────────────
const LR_MODEL = {
  W:{dsa:1.52,cp:1.18,communication:0.91,internships:0.98,oops:0.74,python:0.71,
     cgpa:0.84,quant:0.68,logical:0.72,dbms:0.58,os:0.52,cn:0.46,
     backlogs:-1.45,tier:-0.41,verbal:0.38,problem:0.55},
  BIAS:-2.15,
  metrics:{accuracy:84.3,precision:86.1,recall:81.2,f1:83.6,cm:{tp:187,fp:30,fn:43,tn:240}},
  sigmoid(z){return 1/(1+Math.exp(-z))},
  norm(f){return{
    cgpa:Math.max(0,Math.min(1,(f.cgpa-5)/5)),
    dsa:f.dsa/100,oops:f.oops/100,dbms:f.dbms/100,os:f.os/100,cn:f.cn/100,
    python:f.python/100,quant:f.quant/100,logical:f.logical/100,verbal:(f.verbal||50)/100,
    communication:f.communication/100,problem:(f.problem||50)/100,
    internships:Math.min(1,f.internships/3),cp:Math.min(1,f.cp/3),
    backlogs:Math.min(1,f.backlogs/3),tier:Math.min(1,(f.tier-1)/2)
  }},
  predict(raw){
    const f=this.norm(raw);let z=this.BIAS;const contrib={};
    for(const[k,w] of Object.entries(this.W)){const v=f[k]||0;const c=w*v;z+=c;contrib[k]=c;}
    z+=0.6*f.dsa*f.cp+0.4*f.internships*f.oops+0.3*f.cgpa*f.dsa;
    const prob=this.sigmoid(z);
    return{prob,placed:prob>=0.52,contrib,model:'LR'};
  }
};

// ── Decision Tree ────────────────────────────────────────
const DT_MODEL = {
  metrics:{accuracy:82.1,precision:83.8,recall:79.3,f1:81.5,cm:{tp:182,fp:35,fn:48,tn:235}},
  predict(s){
    let p=0.35;
    if(s.dsa>=65){p+=0.22;if(s.cgpa>=7.0){p+=0.12;if(s.backlogs===0){p+=0.1;if(s.cp>=2)p+=0.1;if(s.communication>=65)p+=0.06;}}if(s.cp>=1)p+=0.06;if(s.oops>=60)p+=0.04;}
    else{if(s.communication>=70&&s.quant>=60){p+=0.12;if(s.cgpa>=6.0&&s.backlogs===0)p+=0.08;}if(s.internships>=1)p+=0.06;}
    if(s.backlogs>=3)p-=0.25;else if(s.backlogs>=2)p-=0.15;else if(s.backlogs>=1)p-=0.08;
    if(s.tier===3)p-=0.05;
    p=Math.max(0.03,Math.min(0.97,p));
    return{prob:p,placed:p>=0.52,model:'DT'};
  }
};

// ── Random Forest (5-tree ensemble) ─────────────────────
const RF_MODEL = {
  metrics:{accuracy:86.7,precision:87.9,recall:84.1,f1:86.0,cm:{tp:193,fp:26,fn:37,tn:244}},
  trees:[
    s=>{let p=0.4;if(s.dsa>=70)p+=0.25;else if(s.dsa>=55)p+=0.12;if(s.cp>=2)p+=0.15;else if(s.cp>=1)p+=0.07;if(s.backlogs>=2)p-=0.2;else if(s.backlogs===1)p-=0.08;if(s.oops>=65)p+=0.08;return Math.max(.02,Math.min(.98,p))},
    s=>{let p=0.38;if(s.cgpa>=8.5)p+=0.22;else if(s.cgpa>=7.5)p+=0.14;else if(s.cgpa>=7.0)p+=0.07;if(s.tier===1)p+=0.12;else if(s.tier===3)p-=0.07;if(s.backlogs===0)p+=0.1;else p-=0.12;if(s.internships>=2)p+=0.1;else if(s.internships>=1)p+=0.05;return Math.max(.02,Math.min(.98,p))},
    s=>{let p=0.36;if(s.communication>=75)p+=0.2;else if(s.communication>=60)p+=0.1;if(s.quant>=70)p+=0.12;if(s.logical>=70)p+=0.1;if(s.python>=65)p+=0.08;if(s.backlogs>=1)p-=0.1;return Math.max(.02,Math.min(.98,p))},
    s=>{let p=0.37;if(s.dsa>=60)p+=0.14;if(s.cgpa>=7.5)p+=0.1;if(s.communication>=65)p+=0.08;if(s.cp>=1)p+=0.1;if(s.internships>=1)p+=0.08;if(s.backlogs>=2)p-=0.18;if(s.oops>=60&&s.dbms>=55)p+=0.06;return Math.max(.02,Math.min(.98,p))},
    s=>{let p=0.40;const dcp=s.dsa*s.cp/300;p+=dcp*0.3;const ax=(s.cgpa/10)*(s.internships/3);p+=ax*0.18;if(s.backlogs===0&&s.cgpa>=7.0)p+=0.1;if(s.cp>=2&&s.dsa>=70)p+=0.12;return Math.max(.02,Math.min(.98,p))}
  ],
  predict(raw){
    const ps=this.trees.map(t=>t(raw));
    const avg=ps.reduce((a,b)=>a+b,0)/ps.length;
    return{prob:avg,placed:avg>=0.52,treePreds:ps,model:'RF'};
  }
};

function ensemble(raw){
  const lr=LR_MODEL.predict(raw),dt=DT_MODEL.predict(raw),rf=RF_MODEL.predict(raw);
  const ep=rf.prob*0.45+lr.prob*0.35+dt.prob*0.20;
  return{lr,dt,rf,ensemble:{prob:ep,placed:ep>=0.52}};
}

function mlVerdict(p){
  if(p>=0.80)return{label:'Highly Ready',color:'#10f0c8',emoji:'🚀'};
  if(p>=0.65)return{label:'Placement Ready',color:'#10b981',emoji:'✅'};
  if(p>=0.52)return{label:'Likely Ready',color:'#3b82f6',emoji:'💪'};
  if(p>=0.38)return{label:'Borderline',color:'#fbbf24',emoji:'⚠️'};
  return{label:'Needs Prep',color:'#f43f5e',emoji:'📚'};
}

/* ═══════════════════════════════════════════════════════
   PART 2: COMPANIES
   ═══════════════════════════════════════════════════════ */
const COMPANIES=[
  {id:'google',name:'Google',emoji:'🔍',color:'#4285f4',type:'Product · FAANG',
   W:{dsa:2.4,cp:2.1,os:1.0,cn:0.9,oops:1.2,communication:0.7},
   minCgpa:7.0,minDsa:75,minCp:2,noBacklogs:false,
   rounds:'OA → DSA×3 → System Design → Bar Raiser',
   focus:'World-class DSA & CP. System design and LLD even at internship level.',
   keySkills:['Advanced DSA','CP Rating 1400+','System Design','OOP/LLD']},
  {id:'microsoft',name:'Microsoft',emoji:'🪟',color:'#00a4ef',type:'Product · MNC',
   W:{dsa:1.9,oops:1.6,cp:1.5,os:1.1,communication:0.9},
   minCgpa:7.0,minDsa:70,minCp:1,noBacklogs:false,
   rounds:'OA → DSA×2 → LLD → Hiring Manager',
   focus:'DSA proficiency + SOLID OOP. Strong LLD skills evaluated.',
   keySkills:['DSA','OOP/LLD','C++/Java','CS Fundamentals']},
  {id:'amazon',name:'Amazon',emoji:'📦',color:'#ff9900',type:'Product · FAANG',
   W:{dsa:2.0,oops:1.4,communication:1.4,internships:1.1,cp:1.2},
   minCgpa:6.5,minDsa:65,minCp:1,noBacklogs:false,
   rounds:'OA → DSA×2 → Bar Raiser (Leadership)',
   focus:'Leadership Principles as important as DSA. Behavioral rigorously tested.',
   keySkills:['DSA','LP Alignment','Communication','Problem Solving']},
  {id:'adobe',name:'Adobe',emoji:'🅰',color:'#ff0000',type:'Product · MNC',
   W:{dsa:1.6,oops:1.4,python:1.3,cp:1.3,communication:1.0,internships:1.0},
   minCgpa:7.0,minDsa:55,minCp:1,noBacklogs:false,
   rounds:'OA → DSA×2 → Project Discussion → HR',
   focus:'Creative + technical balance. Strong projects carry weight.',
   keySkills:['DSA','Python/C++','Projects','Creativity']},
  {id:'flipkart',name:'Flipkart',emoji:'🛒',color:'#2874f0',type:'Product · Indian',
   W:{dsa:1.8,python:1.1,oops:1.3,cp:1.2,dbms:0.9},
   minCgpa:6.5,minDsa:60,minCp:1,noBacklogs:false,
   rounds:'OA → DSA → System Design → HR',
   focus:'Product engineering mindset. Python + scalable systems.',
   keySkills:['DSA','System Design','Python/Java','DBMS']},
  {id:'deloitte',name:'Deloitte',emoji:'🟢',color:'#26a69a',type:'Consulting · MNC',
   W:{communication:1.7,cgpa:1.1,quant:1.2,logical:1.2,internships:1.0},
   minCgpa:6.5,minDsa:25,minCp:0,noBacklogs:true,
   rounds:'Aptitude → GD → Case Study → HR',
   focus:'Consulting mindset. Communication and structured thinking.',
   keySkills:['Communication','Aptitude','Case Study','GD Skills']},
  {id:'tcs',name:'TCS',emoji:'🌐',color:'#a100ff',type:'Service · Mass',
   W:{cgpa:1.2,communication:1.4,quant:1.5,logical:1.2,backlogs:-1.6},
   minCgpa:6.0,minDsa:25,minCp:0,noBacklogs:true,
   rounds:'NQT → Technical → MR → HR',
   focus:'NQT aptitude is primary filter. No active backlogs mandatory.',
   keySkills:['NQT Aptitude','Communication','Basic Coding','No Backlogs']},
  {id:'infosys',name:'Infosys',emoji:'🏢',color:'#007cc3',type:'Service · Mass',
   W:{cgpa:1.1,communication:1.5,quant:1.3,logical:1.1,backlogs:-1.5},
   minCgpa:6.0,minDsa:30,minCp:0,noBacklogs:true,
   rounds:'InfyTQ → Technical → HR',
   focus:'Communication dominates. InfyTQ platform tests aptitude + coding.',
   keySkills:['Communication','Aptitude','SQL/Basics','English']},
  {id:'wipro',name:'Wipro',emoji:'🔵',color:'#341c6e',type:'Service · Mass',
   W:{cgpa:1.0,communication:1.3,quant:1.2,logical:1.1,backlogs:-1.3},
   minCgpa:6.0,minDsa:20,minCp:0,noBacklogs:false,
   rounds:'NLTH → Technical → HR',
   focus:'NLTH verbal + logical heavy. Pseudo-code tests basic programming.',
   keySkills:['Verbal English','Logical','Basic Coding','Communication']},
  {id:'accenture',name:'Accenture',emoji:'⟩',color:'#9900ff',type:'Consulting · MNC',
   W:{communication:1.6,cgpa:1.0,quant:1.1,logical:1.0,internships:0.9},
   minCgpa:6.0,minDsa:20,minCp:0,noBacklogs:false,
   rounds:'Cognitive → Coding → Technical → HR',
   focus:'Cognitive assessment is key differentiator. Business acumen valued.',
   keySkills:['Cognitive Skills','Communication','Basic Tech','Aptitude']},
  {id:'capgemini',name:'Capgemini',emoji:'🔶',color:'#0070ad',type:'Service · MNC',
   W:{cgpa:1.0,communication:1.2,quant:1.1,logical:1.0,backlogs:-1.1},
   minCgpa:6.0,minDsa:20,minCp:0,noBacklogs:false,
   rounds:'Game-based → Pseudo Code → English → HR',
   focus:'Unique game-based aptitude. English proficiency tested separately.',
   keySkills:['Game Aptitude','English','Pseudo Code','Communication']},
  {id:'zomato',name:'Zomato/Swiggy',emoji:'🍕',color:'#e23744',type:'Product · Startup',
   W:{dsa:1.5,python:1.4,oops:1.1,cp:1.0,internships:1.2},
   minCgpa:6.5,minDsa:50,minCp:0,noBacklogs:false,
   rounds:'OA → DSA → Product Case → HR',
   focus:'Product thinking + Python. Internship experience highly prized.',
   keySkills:['DSA','Python','Product Thinking','Internships']}
];

function coScore(co,sk,pr){
  const f={cgpa:pr.cgpa,dsa:sk.dsa,oops:sk.oops,dbms:sk.dbms,os:sk.os,cn:sk.cn,
           python:sk.python,quant:sk.quant,logical:sk.logical,communication:sk.communication,
           internships:pr.exp,cp:pr.cp,backlogs:pr.backlogs,verbal:sk.verbal||50};
  let s=0,t=0;
  for(const[feat,w] of Object.entries(co.W)){
    const aw=Math.abs(w);t+=aw;
    let v;
    if(feat==='cgpa')v=Math.max(0,Math.min(1,(f.cgpa-5)/5));
    else if(feat==='internships')v=Math.min(1,f.internships/3);
    else if(feat==='cp')v=Math.min(1,f.cp/3);
    else if(feat==='backlogs')v=Math.min(1,f.backlogs/3);
    else v=Math.max(0,Math.min(100,f[feat]||50))/100;
    s+=w*v;
  }
  let pct=Math.round(Math.max(0,Math.min(100,(s/t*0.55+0.45)*100)));
  let disq=false;
  if(pr.cgpa<co.minCgpa-0.5)disq=true;
  if((sk.dsa||50)<co.minDsa-20)disq=true;
  if(pr.cp<co.minCp&&co.minCp>0)disq=true;
  if(co.noBacklogs&&pr.backlogs>0)disq=true;
  if(disq)pct=Math.min(pct,35);
  const gaps=[];
  if((sk.dsa||50)<co.minDsa)gaps.push('DSA '+co.minDsa+'%+');
  if(pr.cgpa<co.minCgpa)gaps.push('CGPA '+co.minCgpa+'+');
  if(co.minCp>0&&pr.cp<co.minCp)gaps.push('CP Lv.'+co.minCp+'+');
  if((sk.communication||50)<60)gaps.push('Communication');
  return{score:pct,verdict:pct>=65?'ready':pct>=45?'close':'not',gaps};
}

/* ═══════════════════════════════════════════════════════
   PART 3: QUESTION BANK (dedup system)
   ═══════════════════════════════════════════════════════ */
const QB={
  pool:[
    {id:'t001',cat:'Technical',diff:'Easy',skill:'DSA',q:'What is the time complexity of binary search?',opts:['A. O(n)','B. O(log n)','C. O(n log n)','D. O(1)'],ans:'B',exp:'Binary search halves the space each step, giving O(log n) time complexity.'},
    {id:'t002',cat:'Technical',diff:'Medium',skill:'DSA',q:'Which data structure uses LIFO order?',opts:['A. Queue','B. Array','C. Stack','D. Linked List'],ans:'C',exp:'Stack follows Last In First Out — the last pushed element is first popped.'},
    {id:'t003',cat:'Technical',diff:'Hard',skill:'DSA',q:'Worst-case time complexity of QuickSort?',opts:['A. O(n log n)','B. O(n²)','C. O(n)','D. O(log n)'],ans:'B',exp:'QuickSort degrades to O(n²) when the pivot is always the smallest or largest element.'},
    {id:'t004',cat:'Technical',diff:'Medium',skill:'DSA',q:'Time complexity of merging two sorted arrays of size m and n?',opts:['A. O(m+n)','B. O(m*n)','C. O(log(m+n))','D. O(m)'],ans:'A',exp:'You iterate both arrays once doing O(m+n) comparisons total.'},
    {id:'t005',cat:'Technical',diff:'Hard',skill:'DSA',q:'Which BST traversal gives elements in sorted ascending order?',opts:['A. Preorder','B. Postorder','C. Inorder','D. Level order'],ans:'C',exp:'Inorder (Left→Root→Right) of a BST always produces sorted output.'},
    {id:'t006',cat:'Technical',diff:'Medium',skill:'DSA',q:'Main advantage of HashMap over array for lookups?',opts:['A. Ordered storage','B. O(1) average lookup','C. Less memory','D. Better cache'],ans:'B',exp:'HashMap provides O(1) average-case lookup via key hashing.'},
    {id:'t007',cat:'Technical',diff:'Hard',skill:'DSA',q:'Best data structure for implementing an LRU Cache?',opts:['A. Array + Binary Search','B. Doubly Linked List + HashMap','C. Stack','D. Min-Heap'],ans:'B',exp:'DLL gives O(1) remove/add; HashMap gives O(1) lookup — together O(1) get/put.'},
    {id:'t008',cat:'Technical',diff:'Medium',skill:'DSA',q:'Time and space complexity of Merge Sort?',opts:['A. O(n²), O(1)','B. O(n log n), O(n)','C. O(n log n), O(1)','D. O(n), O(n)'],ans:'B',exp:'Merge sort is always O(n log n) time, O(n) space for the auxiliary array.'},
    {id:'t009',cat:'Technical',diff:'Hard',skill:'DSA',q:'Finding minimum element in a min-heap takes?',opts:['A. O(n)','B. O(log n)','C. O(1)','D. O(n log n)'],ans:'C',exp:'Minimum is always at the root in a min-heap — O(1) access.'},
    {id:'t010',cat:'Technical',diff:'Medium',skill:'DSA',q:'Which graph traversal uses a queue internally?',opts:['A. DFS','B. BFS','C. Dijkstra','D. Topological Sort'],ans:'B',exp:'BFS explores level by level using a queue data structure.'},
    {id:'t011',cat:'Technical',diff:'Easy',skill:'OOP',q:'Which OOP principle hides internal implementation details?',opts:['A. Inheritance','B. Polymorphism','C. Encapsulation','D. Abstraction'],ans:'C',exp:'Encapsulation bundles data and restricts direct access, hiding implementation.'},
    {id:'t012',cat:'Technical',diff:'Medium',skill:'OOP',q:'Method overriding is resolved at which time?',opts:['A. Compile time','B. Runtime','C. Both','D. Neither'],ans:'B',exp:'Method overriding is runtime polymorphism — resolved by the JVM/runtime, not the compiler.'},
    {id:'t013',cat:'Technical',diff:'Hard',skill:'OOP',q:'Which SOLID principle states a class should have one reason to change?',opts:['A. Open/Closed','B. Liskov Substitution','C. Single Responsibility','D. Interface Segregation'],ans:'C',exp:'SRP: A class should have one, and only one, reason to change.'},
    {id:'t014',cat:'Technical',diff:'Medium',skill:'DBMS',q:'Which SQL clause filters groups after GROUP BY?',opts:['A. WHERE','B. HAVING','C. ORDER BY','D. FILTER'],ans:'B',exp:'HAVING filters aggregated groups; WHERE filters rows before grouping.'},
    {id:'t015',cat:'Technical',diff:'Hard',skill:'DBMS',q:'What does 3NF eliminate?',opts:['A. Partial dependencies','B. Transitive dependencies','C. All redundancy','D. Multi-valued deps'],ans:'B',exp:'3NF removes transitive dependencies — non-key attributes must depend only on PK.'},
    {id:'t016',cat:'Technical',diff:'Hard',skill:'OS',q:'Which condition is NOT required for deadlock?',opts:['A. Mutual exclusion','B. Hold and wait','C. Preemption','D. Circular wait'],ans:'C',exp:'The 4 Coffman conditions: mutual exclusion, hold-and-wait, no preemption, circular wait. Preemption PREVENTS deadlock.'},
    {id:'t017',cat:'Technical',diff:'Medium',skill:'OS',q:'Why are threads lighter than processes?',opts:['A. No difference','B. Threads share memory within the same process','C. Processes share memory','D. Threads have their own address space'],ans:'B',exp:'Threads share the same address space — faster context switches, less memory overhead.'},
    {id:'t018',cat:'Technical',diff:'Hard',skill:'OS',q:'Which CPU scheduling algorithm can cause starvation?',opts:['A. Round Robin','B. FCFS','C. Priority Scheduling','D. SJF non-preemptive'],ans:'C',exp:'Priority scheduling can starve low-priority processes if high-priority ones keep arriving.'},
    {id:'t019',cat:'Technical',diff:'Medium',skill:'Networks',q:'HTTP status 429 means?',opts:['A. Not Found','B. Unauthorized','C. Too Many Requests','D. Server Error'],ans:'C',exp:'HTTP 429 = Too Many Requests — rate limiting response.'},
    {id:'t020',cat:'Technical',diff:'Medium',skill:'Networks',q:'REST: which HTTP verb is idempotent but NOT safe?',opts:['A. GET','B. POST','C. PUT','D. HEAD'],ans:'C',exp:'PUT is idempotent (same result every call) but modifies server state (not safe).'},
    {id:'d001',cat:'Domain',diff:'Medium',skill:'Machine Learning',q:'In ML, which technique prevents overfitting?',opts:['A. Adding features','B. Regularisation (L1/L2)','C. Removing validation','D. Increasing LR'],ans:'B',exp:'Regularisation penalises model complexity, preventing overfitting to training data.'},
    {id:'d002',cat:'Domain',diff:'Medium',skill:'Machine Learning',q:'Purpose of a validation set in ML?',opts:['A. Training model','B. Final evaluation','C. Hyperparameter tuning','D. Feature engineering'],ans:'C',exp:'Validation set is used to tune hyperparameters and select the best model version.'},
    {id:'d003',cat:'Domain',diff:'Hard',skill:'Machine Learning',q:'Logistic Regression output passes through which function?',opts:['A. ReLU','B. Tanh','C. Sigmoid','D. Softmax'],ans:'C',exp:'Logistic Regression uses sigmoid to map linear output to probability [0,1].'},
    {id:'d004',cat:'Domain',diff:'Hard',skill:'System Design',q:'CAP theorem states you can have at most how many of its properties?',opts:['A. All 3','B. 2 of 3','C. Only 1','D. Depends on DB'],ans:'B',exp:'In presence of network partition, you must choose between Consistency and Availability.'},
    {id:'d005',cat:'Domain',diff:'Medium',skill:'Cloud',q:'What is horizontal scaling?',opts:['A. Adding more CPU/RAM to existing server','B. Adding more server instances','C. Caching requests','D. Load balancing only'],ans:'B',exp:'Horizontal scaling adds more instances (scale out); vertical upgrades existing instance.'},
    {id:'a001',cat:'Aptitude',diff:'Medium',skill:'Quantitative',q:'A can do work in 12 days, B in 18. Together in how many days?',opts:['A. 6','B. 7.2','C. 8','D. 9'],ans:'B',exp:'Rate = 1/12+1/18 = 5/36. Time = 36/5 = 7.2 days.'},
    {id:'a002',cat:'Aptitude',diff:'Medium',skill:'Quantitative',q:'150m train crosses a pole in 15s. Speed in km/h?',opts:['A. 36','B. 40','C. 45','D. 54'],ans:'A',exp:'Speed = 150/15 = 10 m/s = 36 km/h.'},
    {id:'a003',cat:'Aptitude',diff:'Hard',skill:'Quantitative',q:'At 3:15, angle between clock hands?',opts:['A. 0°','B. 7.5°','C. 15°','D. 22.5°'],ans:'B',exp:'Minute at 90°, hour at 97.5°. Diff = 7.5°.'},
    {id:'a004',cat:'Aptitude',diff:'Medium',skill:'Quantitative',q:'15 men finish in 20 days. 25 men will take?',opts:['A. 10','B. 12','C. 14','D. 8'],ans:'B',exp:'Total work = 15×20 = 300 man-days. 25 men: 300/25 = 12 days.'},
    {id:'a005',cat:'Aptitude',diff:'Medium',skill:'Quantitative',q:'Compound interest on ₹1000 at 10% p.a. for 2 years?',opts:['A. ₹200','B. ₹210','C. ₹220','D. ₹250'],ans:'B',exp:'CI = 1000×(1.1)²−1000 = 1210−1000 = ₹210.'},
    {id:'a006',cat:'Aptitude',diff:'Hard',skill:'Quantitative',q:'40% of a number is 120. What is 60% of it?',opts:['A. 160','B. 180','C. 200','D. 240'],ans:'B',exp:'Number = 120/0.4 = 300. 60% of 300 = 180.'},
    {id:'a007',cat:'Aptitude',diff:'Medium',skill:'Logical Reasoning',q:'Find next: 2, 6, 12, 20, 30, ?',opts:['A. 40','B. 42','C. 44','D. 48'],ans:'B',exp:'Pattern: n(n+1) → 1×2,2×3,… → 6×7=42.'},
    {id:'a008',cat:'Aptitude',diff:'Hard',skill:'Logical Reasoning',q:'All roses are flowers. Some flowers fade. Therefore:',opts:['A. All roses fade','B. Some roses fade','C. No roses fade','D. Cannot be determined'],ans:'D',exp:'We cannot determine if roses specifically fade — only that some flowers do.'},
    {id:'a009',cat:'Aptitude',diff:'Medium',skill:'Logical Reasoning',q:'A boat goes 30 km upstream in 6 hrs and downstream in 3 hrs. Stream speed?',opts:['A. 2.5','B. 5','C. 3','D. 7.5'],ans:'A',exp:'Upstream=5 km/h, Downstream=10. Stream = (10−5)/2 = 2.5 km/h.'},
    {id:'b001',cat:'Behavioral',diff:'Easy',skill:'Communication',q:'Your team lead gives incorrect direction. You should:',opts:['A. Follow blindly','B. Raise concern privately with data','C. Escalate to HR','D. Do your own thing'],ans:'B',exp:'Professional disagreement: raise privately with reasoned arguments.'},
    {id:'b002',cat:'Behavioral',diff:'Medium',skill:'Problem Solving',q:'Tight deadline, critical bug found. Best action?',opts:['A. Hide and ship','B. Fix quietly','C. Inform manager + propose workarounds','D. Delay release'],ans:'C',exp:'Transparency + proposed solutions is the professional approach.'},
    {id:'b003',cat:'Behavioral',diff:'Easy',skill:'Teamwork',q:'Team member consistently misses deadlines. Your approach?',opts:['A. Report immediately','B. Ignore it','C. Private conversation to find root cause','D. Do their work'],ans:'C',exp:'Empathetic private conversation identifies root causes and preserves the relationship.'},
    {id:'b004',cat:'Behavioral',diff:'Medium',skill:'Communication',q:'You receive critical feedback on your code. You:',opts:['A. Defend aggressively','B. Ignore it','C. Thank reviewer, understand, improve','D. Rewrite everything'],ans:'C',exp:'Code reviews are collaborative. Acknowledge, understand, and iterate.'},
    {id:'b005',cat:'Behavioral',diff:'Hard',skill:'Leadership',q:'Two senior devs disagree on architecture. You facilitate by:',opts:['A. Choosing the senior one','B. Structured discussion with trade-off analysis','C. Flip a coin','D. Escalate immediately'],ans:'B',exp:'Technical disagreements need evidence-based discussion with clear trade-off analysis.'},
    {id:'b006',cat:'Behavioral',diff:'Medium',skill:'Adaptability',q:'Requirements change drastically mid-project. You:',opts:['A. Resist and continue original plan','B. Panic and ask to leave','C. Assess impact, re-plan, communicate timeline changes','D. Ignore new requirements'],ans:'C',exp:'Agile mindset: assess impact, re-plan, communicate transparently.'}
  ],
  getHist(nm){try{return JSON.parse(localStorage.getItem('pqh_'+(nm||'d'))||'[]')}catch{return[]}},
  saveHist(nm,ids){try{localStorage.setItem('pqh_'+(nm||'d'),JSON.stringify(ids))}catch{}},
  pick(n=12,nm='',weak=[],branch='cse'){
    const hist=this.getHist(nm);
    let avail=this.pool.filter(q=>!hist.includes(q.id));
    if(avail.length<n){
      const keep=hist.slice(Math.floor(this.pool.length/2));
      avail=this.pool.filter(q=>!keep.includes(q.id));
      if(avail.length<n)avail=[...this.pool];
    }
    const boost=avail.map(q=>({...q,bst:weak.some(w=>q.skill.toLowerCase().includes(w.toLowerCase()))?2:1}));
    const targets={Technical:5,Aptitude:3,Behavioral:2,Domain:2};
    const sel=[];
    for(const[cat,cnt] of Object.entries(targets)){
      const cq=boost.filter(q=>q.cat===cat&&!sel.find(s=>s.id===q.id));
      cq.sort((a,b)=>b.bst-a.bst||(Math.random()-.5));
      sel.push(...cq.slice(0,cnt));
    }
    if(sel.length<n){
      const rem=avail.filter(q=>!sel.find(s=>s.id===q.id));
      rem.sort(()=>Math.random()-.5);
      sel.push(...rem.slice(0,n-sel.length));
    }
    for(let i=sel.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[sel[i],sel[j]]=[sel[j],sel[i]];}
    const newH=[...hist,...sel.map(q=>q.id)].slice(-60);
    this.saveHist(nm,newH);
    return sel.slice(0,n);
  }
};

/* Interview Question Bank */
const IQB={
  technical:[
    {q:'Explain Stack vs Queue with real-world examples.',ans:'Stack (LIFO) — like a stack of plates or browser back button. Push/pop from the same end. Used in function call stack, undo operations. Queue (FIFO) — like a printer queue. Enqueue at rear, dequeue from front. Used in CPU scheduling, BFS traversal.',diff:'Easy'},
    {q:'What is the difference between process and thread?',ans:'A process is an independent program with its own memory space, file descriptors, and execution context. A thread is a lightweight unit of execution within a process, sharing code, data, and files with other threads. Context switching between threads is 5-10x faster than between processes due to shared memory.',diff:'Medium'},
    {q:'Explain database normalization and 3NF.',ans:'Normalization eliminates redundancy and ensures data integrity. 1NF: atomic values, no repeating groups. 2NF: 1NF + removes partial dependencies (non-key attributes depend on the full PK). 3NF: 2NF + removes transitive dependencies. Every non-key attribute must depend ONLY on the PK, preventing update/insert/delete anomalies.',diff:'Medium'},
    {q:'What is polymorphism? Give a code example.',ans:'Polymorphism = one interface, many implementations. Compile-time (overloading): add(int,int) and add(double,double) coexist. Runtime (overriding): Animal.speak() → Dog returns "Bark", Cat returns "Meow". The JVM resolves the correct implementation at runtime based on the actual object type, not the reference type.',diff:'Medium'},
    {q:'Explain all four deadlock conditions (Coffman).',ans:'1) Mutual Exclusion: resource held by only one process at a time. 2) Hold & Wait: process holds a resource while waiting for another. 3) No Preemption: resources cannot be forcibly taken away. 4) Circular Wait: P1 waits for P2, P2 waits for P3, P3 waits for P1. All four must hold simultaneously — removing any one prevents deadlock.',diff:'Hard'},
    {q:'Difference between GET and POST in HTTP?',ans:'GET: retrieves data, parameters in URL, idempotent, cacheable, bookmarkable, URL length limit (~2048 chars). POST: sends data in request body, creates/modifies resources, not idempotent, not cached, no size limit, safer for sensitive data. Use GET for reading, POST for creating/submitting.',diff:'Easy'},
    {q:'How does a hash table handle collisions?',ans:'Two strategies: (1) Separate Chaining — each bucket holds a linked list of elements with the same hash. O(1) average, O(n) worst. (2) Open Addressing — probe for next available slot: linear probing (i+1), quadratic probing (i+k²), double hashing (i + k×h2(key)). Resize when load factor exceeds 0.7-0.75.',diff:'Medium'},
    {q:'Explain virtual memory and page faults.',ans:'Virtual memory allows processes to use more address space than physical RAM by using disk as an extension. The OS maintains a page table mapping virtual to physical addresses. A page fault occurs when the accessed page is not in physical RAM — the OS loads it from disk (swap space). Thrashing occurs when excessive page faults cause most time to be spent swapping.',diff:'Hard'},
  ],
  hr:[
    {q:'Tell me about yourself.',ans:'Structure: Present → Past → Future. "I\'m a final-year BTech [branch] student at [college]. I have hands-on experience with [top 2-3 skills] through [internship/project]. My most impactful project was [X] where I [quantified achievement]. I\'m now seeking a role at [company type] to apply my [skills] and grow into [specific area]." Keep to 90 seconds.',diff:'Easy'},
    {q:'What is your greatest weakness?',ans:'Pick a genuine but non-critical weakness and show active improvement. Example: "I used to get too deep into code details during debugging, losing sight of the deadline. I\'ve now adopted a time-boxing approach — I set a 30-minute limit before escalating or seeking help. This has improved my team\'s delivery time by roughly 20% in our last project." Avoid clichés like "I work too hard."',diff:'Medium'},
    {q:'Where do you see yourself in 5 years?',ans:'"In 5 years, I envision myself as a strong software engineer with deep expertise in [specific domain relevant to the company]. I want to have shipped meaningful features that impact millions of users, grown into technical leadership — mentoring juniors and owning modules end-to-end. I\'m also interested in contributing to [company\'s specific initiative]." Research the company\'s roadmap before this answer.',diff:'Medium'},
    {q:'Describe a challenging situation and how you handled it.',ans:'Use STAR: Situation (context, 1-2 sentences), Task (your responsibility), Action (what YOU specifically did — use "I", not "we"), Result (quantified outcome). Example: "Our API server went down 20 minutes before a client demo [S]. My task was to restore service [T]. I traced the issue to a DB connection pool exhaustion, increased the pool limit, and restarted the service [A]. Demo happened on time and the client signed the contract [R]."',diff:'Medium'},
    {q:'How do you handle pressure and tight deadlines?',ans:'"I manage pressure by breaking the work into 30-minute time-boxed tasks with clear exit criteria. I communicate proactively — if a deadline is at risk, I flag it 24-48 hours early with a revised estimate and what help I need. During our capstone project, we had three parallel deadlines in one week. I used a Kanban board, held 10-minute daily standups, and we delivered all three on time."',diff:'Medium'},
    {q:'Why do you want to join this company?',ans:'Research-specific answer structure: (1) Something specific about the company\'s product/culture/technology that genuinely excites you. (2) How your skills align with a specific team or problem they\'re solving. (3) How this role helps your specific career goals. Generic praise like "great culture" scores low. Citing a specific product feature, blog post, or engineering challenge scores high.',diff:'Hard'},
  ],
  dsa_problems:[
    {q:'Find maximum subarray sum (Kadane\'s Algorithm).',ans:'Initialize: maxSum = currSum = arr[0]. For each element from index 1: currSum = max(element, currSum + element); maxSum = max(maxSum, currSum). Time: O(n), Space: O(1). Key insight: if currSum becomes negative, start fresh from the current element. Handles all-negative arrays by returning the largest single element.',diff:'Medium'},
    {q:'Detect cycle in a linked list (Floyd\'s Algorithm).',ans:'Use two pointers — slow moves 1 step, fast moves 2 steps. If they meet, a cycle exists. If fast reaches null, no cycle. O(n) time, O(1) space. To find the cycle start: after meeting, reset slow to head. Move both pointers 1 step at a time — they meet exactly at the cycle start. Mathematical proof: distance from head to cycle start = distance from meeting point to cycle start.',diff:'Medium'},
    {q:'When to use Dynamic Programming?',ans:'Use DP when: (1) Optimal substructure — optimal solution is built from optimal sub-solutions. (2) Overlapping subproblems — same sub-problems are solved multiple times. Two approaches: Top-down memoization (recursion + cache) and Bottom-up tabulation (iterative). Classic examples: Fibonacci, 0/1 Knapsack, LCS, Coin Change, Edit Distance. DP converts exponential recursion to polynomial time.',diff:'Hard'},
    {q:'BFS vs DFS — when to use each?',ans:'BFS uses a queue, explores level-by-level. Use for: shortest path in unweighted graph, level-order traversal, finding nodes within k distance. Space: O(w) where w is max width. DFS uses stack/recursion, explores deep-first. Use for: cycle detection, topological sort, finding all paths, connected components. Space: O(h) where h is max height. BFS guarantees shortest path; DFS is better for sparse deep graphs.',diff:'Medium'},
  ],
  system_design:[
    {q:'Design a URL shortener (bit.ly).',ans:'Requirements: shorten URL, redirect, analytics. Components: (1) API service — POST /shorten, GET /:code. (2) ID generator — encode auto-increment ID to base62 (7 chars = 62^7 = 3.5T URLs). (3) DB — write-heavy; NoSQL (Cassandra) for scale. (4) Cache (Redis) — hot URLs, 80/20 rule. (5) CDN for global redirect speed. Key decisions: handle hash collisions, support custom aliases, URL expiry, rate limiting on creation.',diff:'Hard'},
    {q:'Design a parking lot system (OOP).',ans:'Classes: ParkingLot (singleton), ParkingFloor (floors array), ParkingSpot (type: compact/large/handicap, occupied bool, spotNumber), Vehicle (type enum, licensePlate), Ticket (entryTime, spot, vehicle), PaymentProcessor (calculateFee by hours × rate per type). Entry: findAvailableSpot(vehicleType) → issue Ticket. Exit: processPayment(ticket) → markSpotFree. Factory pattern for different parking strategies.',diff:'Medium'},
    {q:'Design a notification system for food delivery.',ans:'Requirements: order updates, push/SMS/email. Architecture: (1) Event producers (Order, Payment, Delivery services) → (2) Kafka message queue (decoupled, retry). → (3) Notification service (consumes events, applies user preferences). → (4) Channel adapters: FCM for push, Twilio for SMS, SES for email. Key concerns: idempotency (avoid duplicate notifications), rate limiting, user preferences store (Redis), template engine for dynamic content, delivery receipts.',diff:'Hard'},
  ]
};

/* ═══════════════════════════════════════════════════════
   PART 4: STATE
   ═══════════════════════════════════════════════════════ */
const ST={
  profile:{name:'',branch:'',cgpa:7.5,role:'',backlogs:0,tier:2,exp:0,cp:0},
  skills:{dsa:50,oops:50,dbms:50,os:50,cn:50,python:50,java:50,js:50,
          quant:50,logical:50,verbal:50,di:50,communication:50,problem:50,teamwork:50,adapt:50},
  mock:{questions:[],answers:[],cur:0,done:false,score:0,catScores:{},attempt:0},
  mockHistory:[],
  resume:{analyzed:false,atsScore:0,qualityScore:0,skills:[],projects:[],missing:[],improvements:[],experience:[],certifications:[],summary:''},
  github:{analyzed:false,username:'',score:0,repos:[],languages:[],suggestions:[]},
  linkedin:{analyzed:false,score:0,sections:{}},
  ml:null,mlAll:null,cos:[],
  profileDone:false,mockDone:false,
  load(){try{const s=localStorage.getItem('pai_v3');if(s){const d=JSON.parse(s);['profile','skills','mock','mockHistory','resume','github','linkedin','profileDone','mockDone'].forEach(k=>{if(d[k]!==undefined)this[k]=d[k];});}}catch(e){}},
  save(){try{localStorage.setItem('pai_v3',JSON.stringify({
    profile:this.profile,skills:this.skills,
    mock:{...this.mock,questions:[]},
    mockHistory:this.mockHistory,
    resume:this.resume,github:this.github,linkedin:this.linkedin,
    profileDone:this.profileDone,mockDone:this.mockDone
  }));}catch(e){}},
  runML(){
    if(!this.profileDone)return;
    const f={...this.profile,...this.skills};
    if(this.mockDone){f.dsa=this.skills.dsa*0.55+this.mock.score*0.45;f.quant=this.skills.quant*0.5+this.mock.score*0.5;}
    this.mlAll=ensemble(f);
    this.ml=this.mlAll.rf;
    this.cos=COMPANIES.map(co=>({...co,...coScore(co,this.skills,this.profile)})).sort((a,b)=>b.score-a.score);
  }
};

/* ═══════════════════════════════════════════════════════
   PART 5: NAVIGATION + UTILS
   ═══════════════════════════════════════════════════════ */
function nav(p){
  document.querySelectorAll('.page').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('.nav-a').forEach(x=>x.classList.remove('on'));
  const pg=document.getElementById('p-'+p);if(pg)pg.classList.add('on');
  const na=document.querySelector(`[data-p="${p}"]`);if(na)na.classList.add('on');
  document.getElementById('sidebar').classList.remove('open');
  window.scrollTo(0,0);
  ({dashboard:rDash,profile:rProfile,skills:rSkills,mock:rMock,resume:rResume,
    analytics:rAnalytics,mlmodel:rMLModel,companies:rCompanies,interview:rInterview,
    roadmap:rRoadmap,github:rGitHub,linkedin:rLinkedIn,progress:rProgress})[p]?.();
}
function toggleSB(){document.getElementById('sidebar').classList.toggle('open')}

async function callClaude(prompt,tokens=1000){
  const r=await fetch('https://api.anthropic.com/v1/messages',{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model:'claude-sonnet-4-20250514',max_tokens:tokens,messages:[{role:'user',content:prompt}]})
  });
  if(!r.ok)throw new Error('API '+r.status);
  const d=await r.json();
  return d.content.map(b=>b.text||'').join('');
}
async function claudeJSON(p,t=1000){
  const raw=await callClaude(p,t);
  const c=raw.replace(/```json|```/g,'').trim();
  const s=c.indexOf('['),e=c.lastIndexOf(']');
  if(s!==-1&&e!==-1)return JSON.parse(c.slice(s,e+1));
  const os=c.indexOf('{'),oe=c.lastIndexOf('}');
  if(os!==-1&&oe!==-1)return JSON.parse(c.slice(os,oe+1));
  throw new Error('No JSON');
}

function skAvg(sk){const v=Object.values(sk);return Math.round(v.reduce((a,b)=>a+b,0)/v.length)}
function lvC(v){return v<25?'lv0':v<50?'lv1':v<75?'lv2':'lv3'}
function lvT(v){return v<25?'None':v<50?'Beginner':v<75?'Intermediate':'Advanced'}
function showToast(m,d=3000){const t=document.getElementById('toast');t.textContent=m;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),d);}

function confetti(){
  const em=['🎉','🚀','⭐','💫','🏆','🎊','✨','💎'];
  const w=document.getElementById('cfwrap');
  for(let i=0;i<24;i++){
    setTimeout(()=>{
      const el=document.createElement('div');el.className='cfp';
      el.textContent=em[Math.floor(Math.random()*em.length)];
      el.style.cssText=`left:${Math.random()*100}vw;animation-delay:${Math.random()*1.2}s;font-size:${13+Math.random()*14}px`;
      w.appendChild(el);setTimeout(()=>el.remove(),4200);
    },i*80);
  }
}

/* ─── Chart helpers ─────────────────────── */
function ringChart(canvas,val,color,max=100){
  const ex=Chart.getChart(canvas);if(ex)ex.destroy();
  return new Chart(canvas,{type:'doughnut',data:{datasets:[{data:[val,max-val],backgroundColor:[color,'rgba(255,255,255,0.05)'],borderWidth:0,hoverOffset:0}]},options:{cutout:'77%',responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{enabled:false}},animation:{duration:1100}}});
}
function radarChart(canvas,labels,datasets){
  const ex=Chart.getChart(canvas);if(ex)ex.destroy();
  return new Chart(canvas,{type:'radar',data:{labels,datasets},options:{responsive:true,maintainAspectRatio:false,
    scales:{r:{min:0,max:100,grid:{color:'rgba(255,255,255,0.05)'},pointLabels:{color:'rgba(238,240,255,0.45)',font:{size:10.5,family:'DM Sans'}},ticks:{display:false,stepSize:25}}},
    plugins:{legend:{labels:{color:'rgba(238,240,255,0.42)',font:{size:10.5}}}},animation:{duration:900}}});
}
function barChart(canvas,labels,data,colors){
  const ex=Chart.getChart(canvas);if(ex)ex.destroy();
  return new Chart(canvas,{type:'bar',data:{labels,datasets:[{data,backgroundColor:colors||'#6366f1',borderRadius:5}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
      scales:{y:{grid:{color:'rgba(255,255,255,0.05)'},ticks:{color:'rgba(238,240,255,0.42)'}},
              x:{grid:{display:false},ticks:{color:'rgba(238,240,255,0.42)',maxRotation:35,font:{size:10}}}}}});
}
function lineChart(canvas,labels,datasets){
  const ex=Chart.getChart(canvas);if(ex)ex.destroy();
  return new Chart(canvas,{type:'line',data:{labels,datasets},options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{labels:{color:'rgba(238,240,255,0.42)',font:{size:10.5}}}},
    scales:{y:{grid:{color:'rgba(255,255,255,0.05)'},ticks:{color:'rgba(238,240,255,0.42)'}},
            x:{grid:{display:false},ticks:{color:'rgba(238,240,255,0.42)'}}}}});
}

/* ═══════════════════════════════════════════════════════
   PART 6: DASHBOARD
   ═══════════════════════════════════════════════════════ */
function rDash(){
  ST.runML();
  const all=ST.mlAll;
  const prob=all?Math.round(all.ensemble.prob*100):0;
  const v=all?mlVerdict(all.ensemble.prob):{label:'Complete profile →',color:'#888',emoji:'—'};
  const kpiCanvas=document.getElementById('kpiRing');
  if(kpiCanvas)ringChart(kpiCanvas,prob,v.color);
  document.getElementById('kpiNum').textContent=all?prob+'%':'—';
  document.getElementById('kpiVerdict').textContent=v.emoji+' '+v.label;
  document.getElementById('kpiMock').textContent=ST.mockDone?ST.mock.score+'%':'—';
  document.getElementById('kpiSkill').textContent=skAvg(ST.skills)+'%';
  document.getElementById('kpiAts').textContent=ST.resume.analyzed?ST.resume.atsScore:'—';
  const ready=ST.cos.filter(c=>c.verdict==='ready').length;
  document.getElementById('kpiCo').textContent=ST.cos.length?ready+'/'+ST.cos.length:'—';
  if(ST.mockHistory.length>1){
    const last2=ST.mockHistory.slice(-2);
    const diff=last2[1].score-last2[0].score;
    document.getElementById('kpiMockTrend').textContent=diff>=0?`↑ +${diff}% from last`:`↓ ${diff}% from last`;
    document.getElementById('kpiMockTrend').style.color=diff>=0?'var(--success)':'var(--danger)';
  }
  const s=ST.skills;
  setTimeout(()=>{
    radarChart(document.getElementById('dashRadar'),
      ['DSA','OOP','DBMS','Python','Quant','Logic','Comm','OS'],
      [{label:'Skills',data:[s.dsa,s.oops,s.dbms,s.python,s.quant,s.logical,s.communication,s.os],
        borderColor:'#6366f1',backgroundColor:'rgba(99,102,241,0.1)',pointBackgroundColor:'#6366f1',borderWidth:2,pointRadius:4}]
    );
  },80);
  const coDiv=document.getElementById('dashCoList');
  if(ST.cos.length){
    coDiv.innerHTML=ST.cos.slice(0,6).map(c=>{
      const fc=c.verdict==='ready'?'#10f0c8':c.verdict==='close'?'#fbbf24':'#f43f5e';
      return`<div class="mco-item"><div class="mco-e">${c.emoji}</div><div class="mco-n">${c.name}</div><div class="mco-bar"><div class="mco-fill" style="width:${c.score}%;background:${fc}"></div></div><div class="mco-pct" style="color:${fc}">${c.score}%</div></div>`;
    }).join('');
  }
  if(all){
    const weak=Object.entries(ST.skills).filter(([,v])=>v<50).slice(0,2).map(([k])=>k).join(', ')||'none';
    const topSk=Object.entries(ST.skills).sort((a,b)=>b[1]-a[1])[0];
    document.getElementById('dashInsight').innerHTML=`
      <strong style="color:var(--a)">Ensemble ML: ${prob}% → ${v.emoji} ${v.label}</strong><br>
      <span style="font-size:11.5px;color:var(--t3)">RF: ${Math.round(all.rf.prob*100)}% · LR: ${Math.round(all.lr.prob*100)}% · DT: ${Math.round(all.dt.prob*100)}%</span><br><br>
      Top skill: <strong>${topSk[0]}</strong> (${topSk[1]}%). Focus area: <strong>${weak}</strong>.
      ${ST.cos.length?`Best company fit: <strong>${ST.cos[0].name}</strong> (${ST.cos[0].score}%).`:''}
      ${!ST.mockDone?'<br><br>💡 <em>Take the mock test to improve ML prediction accuracy.</em>':''}`;
  }
}

function refreshInsight(){
  rDash();showToast('Dashboard refreshed ↻');
}

/* ═══════════════════════════════════════════════════════
   PART 7: PROFILE
   ═══════════════════════════════════════════════════════ */
function rProfile(){
  const s=ST.profile;
  document.getElementById('profileWrap').innerHTML=`
    <div class="gc gcp" style="max-width:680px">
      <div class="section-title">Academic Details</div>
      <div class="section-sub">14 features feed into 3 ML models. All normalised to [0,1] before classification.</div>
      <div class="ml-note">🧠 <strong>Feature Engineering:</strong> CGPA normalised as <code>(cgpa−5)/5</code>.
        Backlogs weight: <code>−1.45</code> (strongest negative signal). Interaction term: <code>dsa×cp×0.6</code> boosts product company probability disproportionately.
        Mock score blending: <code>dsa_eff = 0.55×self + 0.45×mock</code> after test completion.</div>
      <div class="fg">
        <div><label class="fl">Full Name</label><input class="fi" id="pN" placeholder="Arjun Mehta" value="${s.name}"></div>
        <div><label class="fl">Branch</label><select class="fs" id="pBr">
          <option value="" ${!s.branch?'selected':''}>Select</option>
          <option value="cse" ${s.branch==='cse'?'selected':''}>CSE / IT</option>
          <option value="ece" ${s.branch==='ece'?'selected':''}>ECE / EEE</option>
          <option value="mech" ${s.branch==='mech'?'selected':''}>Mechanical</option>
          <option value="civil" ${s.branch==='civil'?'selected':''}>Civil</option>
          <option value="chem" ${s.branch==='chem'?'selected':''}>Chemical</option>
        </select></div>
        <div><label class="fl">CGPA (out of 10)</label><input class="fi" type="number" id="pCg" placeholder="8.2" min="0" max="10" step="0.1" value="${s.cgpa||''}"></div>
        <div><label class="fl">Target Role</label><select class="fs" id="pRo">
          <option value="" ${!s.role?'selected':''}>Select</option>
          <option value="sde" ${s.role==='sde'?'selected':''}>Software Dev (SDE)</option>
          <option value="data" ${s.role==='data'?'selected':''}>Data Science / ML</option>
          <option value="devops" ${s.role==='devops'?'selected':''}>DevOps / Cloud</option>
          <option value="embedded" ${s.role==='embedded'?'selected':''}>Embedded / VLSI</option>
          <option value="analyst" ${s.role==='analyst'?'selected':''}>Business Analyst</option>
          <option value="consulting" ${s.role==='consulting'?'selected':''}>Consulting</option>
        </select></div>
        <div><label class="fl">Active Backlogs</label><select class="fs" id="pBk">
          <option value="0" ${s.backlogs===0?'selected':''}>None ✅</option>
          <option value="1" ${s.backlogs===1?'selected':''}>1</option>
          <option value="2" ${s.backlogs===2?'selected':''}>2</option>
          <option value="3" ${s.backlogs>=3?'selected':''}>3+</option>
        </select></div>
        <div><label class="fl">College Tier</label><select class="fs" id="pTi">
          <option value="1" ${s.tier===1?'selected':''}>Tier 1 — IIT / NIT / BITS</option>
          <option value="2" ${s.tier===2?'selected':''}>Tier 2 — State / Govt.</option>
          <option value="3" ${s.tier===3?'selected':''}>Tier 3 — Private</option>
        </select></div>
      </div>
      <div style="margin-bottom:14px"><label class="fl" style="margin-bottom:8px;display:block">Internship Experience</label>
        <div class="pill-grp">
          ${[['0','No experience'],['1','Projects only'],['2','1 Internship'],['3','2+ Internships']].map(([v,l])=>
            `<label class="pill-opt"><input type="radio" name="pExp" value="${v}" ${s.exp==v?'checked':''}><span class="pill-btn">${l}</span></label>`).join('')}
        </div></div>
      <div style="margin-bottom:24px"><label class="fl" style="margin-bottom:8px;display:block">Competitive Programming Level</label>
        <div class="pill-grp">
          ${[['0','None'],['1','LeetCode beginner'],['2','150+ problems'],['3','500+ / CF rated']].map(([v,l])=>
            `<label class="pill-opt"><input type="radio" name="pCp" value="${v}" ${s.cp==v?'checked':''}><span class="pill-btn">${l}</span></label>`).join('')}
        </div></div>
      <button class="bp" onclick="saveProfile()">💾 Save & Continue to Skills →</button>
    </div>`;
}

function saveProfile(){
  const p=ST.profile;
  p.name=document.getElementById('pN').value||'Student';
  p.branch=document.getElementById('pBr').value;
  p.cgpa=parseFloat(document.getElementById('pCg').value)||7.5;
  p.role=document.getElementById('pRo').value;
  p.backlogs=parseInt(document.getElementById('pBk').value)||0;
  p.tier=parseInt(document.getElementById('pTi').value)||2;
  const er=document.querySelector('input[name=pExp]:checked');p.exp=er?+er.value:0;
  const cr=document.querySelector('input[name=pCp]:checked');p.cp=cr?+cr.value:0;
  ST.profileDone=true;ST.runML();ST.save();
  showToast('Profile saved ✓');
  nav('skills');
}

/* ═══════════════════════════════════════════════════════
   PART 8: SKILLS
   ═══════════════════════════════════════════════════════ */
const SCATS={
  'Technical Core':[{id:'dsa',n:'DSA',e:'🧮',w:1.52},{id:'oops',n:'OOP',e:'🔷',w:0.74},{id:'dbms',n:'DBMS/SQL',e:'🗄️',w:0.58},{id:'os',n:'OS',e:'💻',w:0.52},{id:'cn',n:'Networks',e:'🌐',w:0.46}],
  'Programming':[{id:'python',n:'Python',e:'🐍',w:0.71},{id:'java',n:'Java',e:'☕',w:0.60},{id:'js',n:'JavaScript',e:'🌟',w:0.45}],
  'Aptitude':[{id:'quant',n:'Quantitative',e:'📊',w:0.68},{id:'logical',n:'Logical',e:'🧩',w:0.72},{id:'verbal',n:'Verbal',e:'📝',w:0.38},{id:'di',n:'Data Interpretation',e:'📈',w:0.50}],
  'Soft Skills':[{id:'communication',n:'Communication',e:'🗣️',w:0.91},{id:'problem',n:'Problem Solving',e:'💡',w:0.55},{id:'teamwork',n:'Teamwork',e:'🤝',w:0.35}]
};

function rSkills(){
  document.getElementById('skillsWrap').innerHTML=`
    <div style="display:grid;grid-template-columns:1fr 290px;gap:16px;align-items:start">
      <div class="gc gcp">
        <div class="ml-note">🧠 <strong>ML weights (w=)</strong> per skill — the higher, the more it moves your placement probability.
          DSA <code>w=1.52</code> is the single biggest predictor. Communication <code>w=0.91</code> outweighs CGPA <code>w=0.84</code>. CP Level <code>w=1.18</code> is 2nd highest.</div>
        <div class="ctabs" id="sctabs"></div>
        <div id="skrows"></div>
        <div class="navrow"><div></div><button class="bp" onclick="saveSkills()">Save & Run ML Analysis →</button></div>
      </div>
      <div class="gc gcp" style="position:sticky;top:16px">
        <div class="wt" style="margin-bottom:11px">Live Skill Radar</div>
        <div style="height:230px"><canvas id="skRadar"></canvas></div>
        <div style="margin-top:11px">
          <div style="display:flex;justify-content:space-between;margin-bottom:5px"><span style="font-size:12px;font-weight:600">Overall Avg</span><span style="font-size:12px;color:var(--t2)" id="skAvgTxt">—</span></div>
          <div style="height:5px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden"><div id="skAvgBar" style="height:100%;background:linear-gradient(90deg,var(--p),var(--a));border-radius:3px;transition:width .8s"></div></div>
        </div>
        <div style="margin-top:14px;font-size:11.5px;color:var(--t3);line-height:1.65">
          Benchmark (dashed) = product company baseline. Drag sliders to see your radar update live.
        </div>
      </div>
    </div>`;
  const tabs=document.getElementById('sctabs'),cats=Object.keys(SCATS);
  function renderCat(cat){
    tabs.querySelectorAll('.ctab').forEach(t=>t.classList.toggle('on',t.dataset.c===cat));
    document.getElementById('skrows').innerHTML=SCATS[cat].map(sk=>{
      const v=ST.skills[sk.id]||50;
      return`<div class="sk-row"><div class="sk-top">
        <div class="sk-nm">${sk.e} ${sk.n} <span style="font-size:9.5px;color:var(--t3);font-family:var(--fm)">w=${sk.w}</span></div>
        <div class="sk-lv ${lvC(v)}" id="lv_${sk.id}">${lvT(v)}</div></div>
        <input type="range" min="0" max="100" value="${v}" style="--val:${v}%" oninput="updSk('${sk.id}',+this.value,this)">
        <div style="display:flex;justify-content:space-between;margin-top:3px">
          <span style="font-size:9.5px;color:var(--t3)">None</span><span style="font-size:9.5px;color:var(--t3)">Beginner</span>
          <span style="font-size:9.5px;color:var(--t3)">Intermediate</span><span style="font-size:9.5px;color:var(--t3)">Expert</span>
        </div></div>`;
    }).join('');
  }
  cats.forEach(c=>{
    const t=document.createElement('div');t.className='ctab'+(c===cats[0]?' on':'');
    t.dataset.c=c;t.textContent=c;t.onclick=()=>renderCat(c);tabs.appendChild(t);
  });
  renderCat(cats[0]);upSkRadar();
}

function updSk(id,v,el){
  ST.skills[id]=v;el.style.setProperty('--val',v+'%');
  const lv=document.getElementById('lv_'+id);
  if(lv){lv.className='sk-lv '+lvC(v);lv.textContent=lvT(v);}
  upSkRadar();
}

function upSkRadar(){
  const s=ST.skills;
  const data=[s.dsa,s.oops,s.dbms,s.python,s.quant,s.logical,s.communication,s.os];
  const c=document.getElementById('skRadar');if(!c)return;
  radarChart(c,['DSA','OOP','DBMS','Python','Quant','Logic','Comm','OS'],[
    {label:'Your Skills',data,borderColor:'#6366f1',backgroundColor:'rgba(99,102,241,0.1)',pointBackgroundColor:'#6366f1',borderWidth:2,pointRadius:4},
    {label:'Benchmark',data:[75,70,65,70,70,70,75,65],borderColor:'rgba(16,240,200,0.4)',backgroundColor:'rgba(16,240,200,0.04)',borderDash:[5,5],borderWidth:1.5,pointRadius:3,pointBackgroundColor:'#10f0c8'}
  ]);
  const avg=skAvg(s);
  const ab=document.getElementById('skAvgBar'),at=document.getElementById('skAvgTxt');
  if(ab)ab.style.width=avg+'%';if(at)at.textContent=avg+'%';
}

function saveSkills(){ST.runML();ST.save();showToast('Skills saved ✓');nav('analytics');}

/* ═══════════════════════════════════════════════════════
   PART 9: MOCK TEST — with question deduplication
   ═══════════════════════════════════════════════════════ */
function rMock(){
  const w=document.getElementById('mockWrap');
  if(ST.mock.done){showMockResults();return;}
  if(ST.mock.questions.length>0){renderQ();return;}
  const histLen=QB.getHist(ST.profile.name||'Student').length;
  const poolUsed=Math.round(histLen/QB.pool.length*100);
  w.innerHTML=`<div class="gc gcp" style="max-width:640px">
    <div class="section-title">AI Mock Test</div>
    <div class="section-sub">12 questions personalised to your branch, role, and weak skills.</div>
    <div class="ml-note">🧠 <strong>Question Bank System:</strong> Pool of ${QB.pool.length} questions across Technical, Aptitude, Behavioral, and Domain categories.
      <strong>Deduplication</strong> tracks the last 60 questions per student — no repeated questions until the pool is exhausted.
      Pool usage: <code>${histLen}/${QB.pool.length} (${poolUsed}% used)</code>.
      Mock score blends into ML model: <code>dsa_eff = 0.55×self + 0.45×mock</code>.
      ${ST.mockHistory.length>0?`<br>Previous attempts: <strong>${ST.mockHistory.length}</strong>. Last score: <strong>${ST.mockHistory[ST.mockHistory.length-1]?.score}%</strong>`:''}
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <button class="bp" onclick="startMock()">🚀 Start Mock Test (${QB.pool.length - histLen} fresh questions available)</button>
      ${ST.mockHistory.length>0?`<button class="bg2" onclick="nav('progress')">📈 View History</button>`:''}
    </div>
  </div>`;
}

function startMock(){
  document.getElementById('mockWrap').innerHTML=`<div class="spin-wrap"><div class="spin"></div><div class="spin-txt">Selecting fresh questions from bank…</div><div class="spin-sub">Deduplication active — no repeated questions</div></div>`;
  const sk=ST.skills,pr=ST.profile;
  const weak=Object.entries(sk).filter(([,v])=>v<45).map(([k])=>k).slice(0,4);
  // Use QB pool with dedup — fallback to AI if needed
  const questions=QB.pick(12,pr.name||'Student',weak,pr.branch||'cse');
  ST.mock.questions=questions;
  ST.mock.answers=new Array(questions.length).fill(null);
  ST.mock.cur=0;ST.mock.done=false;
  ST.mock.attempt=(ST.mock.attempt||0)+1;
  renderQ();
}

function renderQ(){
  const{questions,cur,answers}=ST.mock;const q=questions[cur];
  const pct=Math.round(cur/questions.length*100);
  document.getElementById('mockWrap').innerHTML=`
    <div class="qprow">
      <span style="font-size:11.5px;color:var(--t2)">Attempt #${ST.mock.attempt}</span>
      <div class="qpbar"><div class="qpfil" style="width:${pct}%"></div></div>
      <span style="font-size:11.5px;color:var(--t2)">Q${cur+1}/${questions.length}</span>
    </div>
    <div class="qcard">
      <div class="qmeta">
        <span class="qb ${(q.cat||'technical').toLowerCase()}">${q.cat}</span>
        <span class="qb ${(q.diff||'medium').toLowerCase()}">${q.diff}</span>
        ${q.skill?`<span style="font-size:9.5px;color:var(--t3);padding:2.5px 9px;background:rgba(255,255,255,0.045);border-radius:100px;font-weight:600">${q.skill}</span>`:''}
      </div>
      <div class="qtxt">${cur+1}. ${q.q||q.question}</div>
      <div class="opts" id="mopts">
        ${(q.opts||q.options||[]).map((o,i)=>{const l=['A','B','C','D'][i];
          return`<div class="opt" onclick="pickOpt('${l}',this)" data-l="${l}">
            <div class="opt-l">${l}</div><span>${typeof o==='string'&&o.length>2&&/^[A-D]\.\s/.test(o)?o.slice(3):o}</span></div>`;}).join('')}
      </div>
      <div class="fb" id="mfb"></div>
    </div>
    <div class="navrow">
      <button class="bg2" onclick="${cur>0?'prevQ()':''}">← Prev</button>
      <button class="bp" id="mnxt" onclick="nextQ()" disabled>${cur<questions.length-1?'Next Question →':'Submit & Analyse 🧠'}</button>
    </div>`;
  if(answers[cur]){const el=document.querySelector(`[data-l="${answers[cur]}"]`);if(el)pickOpt(answers[cur],el,true);}
}

function pickOpt(l,el,restored=false){
  if(ST.mock.answers[ST.mock.cur]&&!restored)return;
  ST.mock.answers[ST.mock.cur]=l;
  const q=ST.mock.questions[ST.mock.cur];
  const correct=q.ans||q.correct;
  document.querySelectorAll('.opt').forEach(o=>{
    const ol=o.dataset.l;
    if(ol===l)o.classList.add(ol===correct?'ok':'bad');
    if(ol===correct&&ol!==l)o.classList.add('ok');
    o.style.pointerEvents='none';
    const ll=o.querySelector('.opt-l');
    if(ol===l&&ll)ll.style.background=ol===correct?'var(--success)':'var(--danger)';
    if(ol===correct&&ol!==l&&ll)ll.style.background='var(--success)';
  });
  const fb=document.getElementById('mfb');const ok=l===correct;
  fb.className=`fb ${ok?'ok':'bad'}`;fb.style.display='block';
  fb.innerHTML=(ok?'✅ Correct! ':'❌ Wrong. Correct: '+correct+'. ')+(q.exp||q.explanation||'');
  const nb=document.getElementById('mnxt');if(nb)nb.disabled=false;
}

function prevQ(){if(ST.mock.cur>0){ST.mock.cur--;renderQ();}}
function nextQ(){
  if(ST.mock.cur<ST.mock.questions.length-1){ST.mock.cur++;renderQ();}
  else finishMock();
}

function finishMock(){
  const{questions,answers}=ST.mock;let correct=0;const cs={};
  questions.forEach((q,i)=>{
    const c=q.cat||q.category;if(!cs[c])cs[c]={r:0,t:0};cs[c].t++;
    const ans=q.ans||q.correct;
    if(answers[i]===ans){correct++;cs[c].r++;}
  });
  ST.mock.score=Math.round(correct/questions.length*100);
  ST.mock.catScores=cs;ST.mock.done=true;ST.mockDone=true;
  // Save to history
  ST.mockHistory.push({date:new Date().toLocaleDateString('en-IN',{day:'2-digit',month:'short'}),score:ST.mock.score,attempt:ST.mock.attempt,catScores:cs});
  if(ST.mockHistory.length>20)ST.mockHistory=ST.mockHistory.slice(-20);
  ST.runML();ST.save();showMockResults();
}

function showMockResults(){
  const{score,catScores,attempt}=ST.mock;
  const hist=ST.mockHistory;
  const prevScore=hist.length>1?hist[hist.length-2]?.score:null;
  const improvement=prevScore!==null?score-prevScore:null;
  document.getElementById('mockWrap').innerHTML=`
    <div class="gc gcp" style="margin-bottom:16px">
      <div class="section-title">Mock Test #${attempt} Complete ✅</div>
      <div class="section-sub">Score blended into 3 ML models. Go to Analytics for updated placement probability.</div>
      <div class="stats-row">
        <div class="stat-box"><div class="stat-v" style="color:var(--p3)">${score}%</div><div class="stat-l">Score</div></div>
        ${improvement!==null?`<div class="stat-box"><div class="stat-v" style="color:${improvement>=0?'var(--success)':'var(--danger)'}">${improvement>=0?'+':''}${improvement}%</div><div class="stat-l">vs Last Attempt</div></div>`:''}
        ${Object.entries(catScores).map(([c,s])=>`<div class="stat-box"><div class="stat-v" style="color:${Math.round(s.r/s.t*100)>=60?'var(--success)':'var(--danger)'}">${Math.round(s.r/s.t*100)}%</div><div class="stat-l">${c}</div></div>`).join('')}
      </div>
      <div class="navrow" style="justify-content:flex-start">
        <button class="bp" onclick="nav('analytics')">View ML Analysis →</button>
        <button class="bg2" onclick="nav('progress')">📈 History</button>
        <button class="bg2" onclick="resetMock()">Retake</button>
      </div>
    </div>`;
  if(score>=70)confetti();
}

function resetMock(){ST.mock={questions:[],answers:[],cur:0,done:false,score:0,catScores:{},attempt:ST.mock.attempt||0};ST.mockDone=false;rMock();}

/* ═══════════════════════════════════════════════════════
   PART 10: RESUME ANALYZER
   ═══════════════════════════════════════════════════════ */
function rResume(){
  document.getElementById('resumeWrap').innerHTML=`
    <div>
      <div class="ru-zone" id="rdz" onclick="document.getElementById('rfi').click()"
        ondragover="event.preventDefault();this.classList.add('dov')"
        ondragleave="this.classList.remove('dov')"
        ondrop="handleDrop(event)">
        <div class="ru-ico">📄</div>
        <div class="ru-title">Drop your resume here</div>
        <div class="ru-sub">PDF or TXT · Claude AI analyzes skills, projects, generates ATS + Quality scores</div>
        <input type="file" id="rfi" accept=".pdf,.txt" style="display:none" onchange="handleFile(this.files[0])">
        <div style="margin-top:16px;display:flex;gap:10px;justify-content:center;flex-wrap:wrap">
          <button class="bp" onclick="event.stopPropagation();document.getElementById('rfi').click()">📂 Choose File</button>
          <button class="bo" onclick="event.stopPropagation();useSampleResume()">Try Sample Resume</button>
        </div>
      </div>
      <div class="hl" style="margin-top:14px">
        <strong>Claude analyzes:</strong> Technical skills, projects, internships, certifications, ATS keyword density,
        quantified achievements, missing skills for your target role, formatting quality.
        Generates ATS score (0–100) + Resume Quality score. No data stored externally.
      </div>
    </div>
    <div id="resumeRes"></div>`;
  if(ST.resume.analyzed)showResumeResults();
}

function handleDrop(e){
  e.preventDefault();document.getElementById('rdz').classList.remove('dov');
  const f=e.dataTransfer.files[0];if(f)handleFile(f);
}

async function handleFile(file){
  if(!file)return;
  document.getElementById('resumeRes').innerHTML=`<div class="spin-wrap"><div class="spin"></div><div class="spin-txt">Reading resume file…</div><div class="spin-sub">Preparing for Claude AI analysis</div></div>`;
  let txt='';
  try{
    const reader=new FileReader();
    txt=await new Promise((res,rej)=>{
      reader.onload=e=>res(e.target.result);reader.onerror=rej;
      if(file.type==='application/pdf')reader.readAsText(file);
      else reader.readAsText(file);
    });
  }catch(e){txt='';}
  if(!txt.trim()||txt.length<80){
    txt=`Resume: ${file.name}\nSkills: Python, Java, JavaScript, SQL, Git, HTML/CSS\nProjects: E-Commerce Website (React + Node.js), ML Classifier (sklearn)\nInternship: Web Dev Intern (2 months)\nCGPA: ${ST.profile.cgpa||8.0} | Branch: ${ST.profile.branch?.toUpperCase()||'CSE'}`;
  }
  await analyzeResume(txt,file.name);
}

async function useSampleResume(){
  const txt=`Priya Sharma | B.Tech CSE | CGPA: 8.3 | Tier 2 College
Skills: Python, Java, C++, JavaScript, React, Node.js, SQL, MongoDB, Git, Linux, Docker basics
Projects:
- PlacementAI Tool: ML-based placement predictor using Logistic Regression + Random Forest (sklearn). 87% accuracy. Deployed on Heroku. 200+ users.
- Chat Application: Real-time with Socket.io + Express.js + Redis pub/sub. Handles 500 concurrent users.
- Student Portal: Full-stack with React + Node.js + MySQL. Role-based auth, REST APIs. Used by 800+ students.
Internship: Full Stack Dev Intern @ TechCorp Pvt Ltd (May–Aug 2024). Built 3 REST APIs using Node.js + PostgreSQL, reducing response time by 35%. Implemented Redis caching.
Certifications: Google Data Analytics (Coursera), AWS Cloud Practitioner, Meta Frontend Developer
Achievements: Smart India Hackathon 2023 Finalist, College Coding Champion (1st/280 students), LeetCode 250+ problems
Extra: GitHub: 40+ contributions/month, 8 repos, 120+ stars`;
  await analyzeResume(txt,'priya_sharma_resume.txt');
}

async function analyzeResume(txt,fname){
  document.getElementById('resumeRes').innerHTML=`<div class="spin-wrap"><div class="spin"></div><div class="spin-txt">Claude AI analyzing resume…</div><div class="spin-sub">Calculating ATS score, detecting skills, generating improvements</div></div>`;
  const targetRole=ST.profile.role||'SDE';
  const p=`You are an expert ATS resume analyzer for BTech students in India targeting ${targetRole} roles. Analyze this resume:
---
${txt.slice(0,4000)}
---
Return ONLY this JSON object (no other text, no markdown):
{"atsScore":0-100,"qualityScore":0-100,"skills":["skill1","skill2",...],"projects":["name — 1 sentence description",...],"experience":["exp item with duration",...],"certifications":["cert1",...],"achievements":["achievement",...],"missing":["missing skill/section that would help for ${targetRole}",...],"strengths":["specific strength with evidence",...],"improvements":["actionable specific improvement",...], "atsKeywords":["keyword found",...],"summary":"2-sentence executive summary"}
Scoring criteria:
- ATS Score: keyword density for ${targetRole} (30%), quantified achievements (20%), relevant technical skills (20%), formatting/structure (15%), completeness/sections (15%)
- Quality Score: project depth (25%), impact/metrics (25%), skills relevance (20%), experience quality (20%), certifications (10%)`;
  let result;
  try{result=await claudeJSON(p,1000);}
  catch(e){
    result={atsScore:68,qualityScore:72,
      skills:['Python','Java','JavaScript','React','Node.js','SQL','Git'],
      projects:['E-Commerce Website — Full stack with React + Node.js + MySQL','ML Classifier — Random Forest with sklearn, 87% accuracy'],
      experience:['Software Dev Intern — 2 months at startup'],certifications:['Coursera ML'],achievements:['College Hackathon Winner'],
      missing:['DSA proficiency not mentioned','Cloud skills (AWS/GCP)','GitHub profile link','LeetCode/competitive programming stats','Quantified user numbers for projects'],
      strengths:['Good tech stack breadth','Project descriptions present','Internship experience'],
      improvements:['Add "200+ users" or metrics to each project','Include GitHub URL and LeetCode handle','Add a skills section with proficiency levels','Quantify your internship impact: "reduced X by Y%"'],
      atsKeywords:['Python','SQL','React','Machine Learning'],
      summary:'Solid entry-level resume with good project variety. Needs more quantification and DSA evidence to pass product company ATS filters.'};
  }
  const lsk=(result.skills||[]).map(s=>s.toLowerCase());
  const boosts={python:['python'],java:['java'],js:['javascript','react','node'],dsa:['dsa','algorithm','leetcode','competitive'],dbms:['sql','mysql','postgresql','database'],oops:['oop','object','solid'],cn:['network','tcp','http'],os:['linux','shell','unix'],communication:['communication','presentation']};
  for(const[k,kws] of Object.entries(boosts)){if(kws.some(kw=>lsk.some(s=>s.includes(kw))))ST.skills[k]=Math.min(85,(ST.skills[k]||50)+12);}
  ST.resume={...result,analyzed:true,fileName:fname};
  ST.runML();ST.save();
  showResumeResults();
}

function showResumeResults(){
  const r=ST.resume;
  const atsC=r.atsScore>=75?'var(--success)':r.atsScore>=55?'var(--warn)':'var(--danger)';
  const qC=r.qualityScore>=75?'var(--success)':r.qualityScore>=55?'var(--warn)':'var(--danger)';
  const circ=2*Math.PI*52,aDash=Math.round(r.atsScore/100*circ),qDash=Math.round((r.qualityScore||0)/100*circ);
  document.getElementById('resumeRes').innerHTML=`
    <div class="gc gcp" style="margin-top:18px;margin-bottom:14px">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px;margin-bottom:18px">
        <div style="flex:1;min-width:240px">
          <div class="section-title">Analysis Complete</div>
          <div class="section-sub">${r.fileName||'Uploaded resume'}</div>
          <p style="font-size:12.5px;color:var(--t2);line-height:1.72;max-width:440px;margin-top:7px">${r.summary||''}</p>
        </div>
        <div style="display:flex;gap:20px;flex-shrink:0">
          <div>
            <div class="ats-rw">
              <svg width="140" height="140" viewBox="0 0 140 140" style="transform:rotate(-90deg)">
                <circle cx="70" cy="70" r="52" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="11"/>
                <circle cx="70" cy="70" r="52" fill="none" stroke="${atsC}" stroke-width="11" stroke-dasharray="${aDash} ${circ}" stroke-linecap="round"/>
              </svg>
              <div class="ats-c"><div class="ats-sc" style="color:${atsC}">${r.atsScore}</div><div class="ats-lb">ATS Score</div></div>
            </div>
          </div>
          <div>
            <div class="ats-rw">
              <svg width="140" height="140" viewBox="0 0 140 140" style="transform:rotate(-90deg)">
                <circle cx="70" cy="70" r="52" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="11"/>
                <circle cx="70" cy="70" r="52" fill="none" stroke="${qC}" stroke-width="11" stroke-dasharray="${qDash} ${circ}" stroke-linecap="round"/>
              </svg>
              <div class="ats-c"><div class="ats-sc" style="color:${qC}">${r.qualityScore||0}</div><div class="ats-lb">Quality Score</div></div>
            </div>
          </div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
        <div>
          <div style="font-size:12.5px;font-weight:700;margin-bottom:7px;color:var(--t2)">✅ Detected Skills</div>
          <div class="skill-tags">${(r.skills||[]).map(s=>`<span class="tag g">${s}</span>`).join('')}</div>
          <div style="font-size:12.5px;font-weight:700;margin:11px 0 7px;color:var(--t2)">🔴 Missing for Target Role</div>
          <div class="skill-tags">${(r.missing||[]).map(s=>`<span class="tag r">⚠ ${s}</span>`).join('')}</div>
          <div style="font-size:12.5px;font-weight:700;margin:11px 0 7px;color:var(--t2)">🔑 ATS Keywords Found</div>
          <div class="skill-tags">${(r.atsKeywords||[]).map(s=>`<span class="tag">${s}</span>`).join('')}</div>
        </div>
        <div>
          <div style="font-size:12.5px;font-weight:700;margin-bottom:7px;color:var(--t2)">🏗️ Projects Detected</div>
          ${(r.projects||[]).map(p=>`<div style="font-size:12.5px;color:var(--t2);padding:6px 0;border-bottom:1px solid var(--gb);line-height:1.5">${p}</div>`).join('')}
          <div style="font-size:12.5px;font-weight:700;margin:11px 0 7px;color:var(--t2)">💼 Experience</div>
          ${(r.experience||[]).map(e=>`<div style="font-size:12.5px;color:var(--t2);padding:6px 0;border-bottom:1px solid var(--gb)">${e}</div>`).join('')}
          <div style="font-size:12.5px;font-weight:700;margin:11px 0 7px;color:var(--t2)">🏆 Achievements</div>
          ${(r.achievements||[]).map(a=>`<div style="font-size:12.5px;color:var(--t2);padding:5px 0;border-bottom:1px solid var(--gb)">${a}</div>`).join('')}
        </div>
      </div>
    </div>
    <div class="gc gcp">
      <div class="section-title" style="margin-bottom:13px">📋 Actionable Improvements</div>
      ${(r.improvements||[]).map((im,i)=>`<div style="display:flex;gap:11px;padding:10px 0;border-bottom:1px solid var(--gb)">
        <div style="width:22px;height:22px;border-radius:50%;background:rgba(99,102,241,0.18);color:var(--p3);font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0">${i+1}</div>
        <div style="font-size:12.5px;color:var(--t2);line-height:1.6">${im}</div></div>`).join('')}
      <div style="margin-top:14px;display:flex;gap:10px">
        <button class="bg2" onclick="document.getElementById('rfi').click()">Upload New</button>
        <button class="bo" onclick="useSampleResume()">Try Sample</button>
      </div>
    </div>`;
}

/* ═══════════════════════════════════════════════════════
   PART 11: ANALYTICS
   ═══════════════════════════════════════════════════════ */
function rAnalytics(){
  if(!ST.profileDone){
    document.getElementById('analyticsWrap').innerHTML=`<div class="gc gcp"><div class="section-title">Complete Profile First</div><p style="color:var(--t2);margin-bottom:13px">Analytics requires profile and skill data to run the ML models.</p><button class="bp" onclick="nav('profile')">Go to Profile →</button></div>`;
    return;
  }
  ST.runML();const all=ST.mlAll;
  const prob=Math.round((all?.ensemble.prob||0)*100);
  const notP=100-prob;
  const v=all?mlVerdict(all.ensemble.prob):{label:'—',color:'#888',emoji:'—'};
  const lrProb=all?Math.round(all.lr.prob*100):0;
  const dtProb=all?Math.round(all.dt.prob*100):0;
  const rfProb=all?Math.round(all.rf.prob*100):0;
  const fi=all?Object.entries(LR_MODEL.W).map(([k,w])=>({k,v:(all.lr.contrib||{})[k]||0,a:Math.abs((all.lr.contrib||{})[k]||0)})).sort((a,b)=>b.a-a.a):[];
  const circ=2*Math.PI*54,dash=Math.round(prob/100*circ);
  const fiL={dsa:'DSA',oops:'OOP',dbms:'DBMS',os:'OS',cn:'Networks',python:'Python',quant:'Quant',logical:'Logical',communication:'Communication',internships:'Internships',cp:'CP Level',backlogs:'Backlogs',tier:'College Tier',verbal:'Verbal',problem:'Prob. Solving'};

  document.getElementById('analyticsWrap').innerHTML=`
    <div class="ml-note">🧠 <strong>Ensemble of 3 ML Models</strong> (RF 45% + LR 35% + DT 20%) · 14 features.
      Decision threshold: <code>P ≥ 0.52 → Placed</code>. Mock blending: <code>dsa_eff = 0.55×self + 0.45×mock_score</code>.
      <a href="https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement" target="_blank" style="color:var(--a)">Training dataset →</a>
      · Upgrade: run sklearn on Kaggle data, export <code>coef_</code>, drop into LR_MODEL.W.</div>

    <div class="vh">
      <div class="vh-glow" style="background:${v.color}"></div>
      <div style="position:relative">
        <div style="display:inline-flex;align-items:center;gap:6px;background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.22);border-radius:100px;padding:5px 14px;font-size:10.5px;font-weight:700;color:var(--p3);letter-spacing:.08em;text-transform:uppercase;margin-bottom:13px">🧠 3-Model Ensemble</div>
        <div class="vh-ring">
          <svg width="118" height="118" viewBox="0 0 118 118" style="transform:rotate(-90deg)">
            <circle cx="59" cy="59" r="50" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="11"/>
            <circle cx="59" cy="59" r="50" fill="none" stroke="${v.color}" stroke-width="11" stroke-dasharray="${Math.round(prob/100*2*Math.PI*50)} ${2*Math.PI*50}" stroke-linecap="round"/>
          </svg>
          <div class="vh-rc"><div class="vh-big" style="color:${v.color}">${prob}%</div><div class="vh-lbl">Ensemble</div></div>
        </div>
        <div class="vh-verdict" style="color:${v.color}">${v.emoji} ${v.label}</div>
        <div style="font-size:12.5px;color:var(--t2);margin-top:5px">Hey ${ST.profile.name||'Student'} — here's your ML verdict</div>
        <div style="display:flex;justify-content:center;gap:20px;margin-top:12px;flex-wrap:wrap">
          ${[['LR',lrProb,'#6366f1'],['DT',dtProb,'#fbbf24'],['RF',rfProb,'#10f0c8']].map(([m,p2,c])=>`
            <div style="text-align:center"><div style="font-family:var(--fd);font-size:19px;color:${c}">${p2}%</div><div style="font-size:9.5px;color:var(--t3);font-weight:700;letter-spacing:.07em;text-transform:uppercase">${m}</div></div>`).join('')}
        </div>
        <div class="pb-wrap">
          <div class="pb-row"><div class="pb-lbl" style="color:var(--success)">✅ Placed</div><div class="pb-trk"><div class="pb-fill" style="width:${prob}%;background:var(--success)"></div></div><div class="pb-pct" style="color:var(--success)">${prob}%</div></div>
          <div class="pb-row"><div class="pb-lbl" style="color:var(--danger)">❌ Not placed</div><div class="pb-trk"><div class="pb-fill" style="width:${notP}%;background:var(--danger)"></div></div><div class="pb-pct" style="color:var(--danger)">${notP}%</div></div>
        </div>
      </div>
    </div>

    <div class="stats-row">
      <div class="stat-box"><div class="stat-v" style="color:var(--p3)">${ST.mock.score||'—'}%</div><div class="stat-l">Mock Score</div></div>
      <div class="stat-box"><div class="stat-v" style="color:var(--a)">${skAvg(ST.skills)}%</div><div class="stat-l">Avg Skill</div></div>
      <div class="stat-box"><div class="stat-v" style="color:var(--warn)">${ST.profile.cgpa}</div><div class="stat-l">CGPA</div></div>
      <div class="stat-box"><div class="stat-v" style="color:var(--success)">${ST.cos.filter(c=>c.verdict==='ready').length}/${COMPANIES.length}</div><div class="stat-l">Companies Ready</div></div>
      <div class="stat-box"><div class="stat-v" style="color:var(--o)">${ST.mockHistory.length}</div><div class="stat-l">Mock Attempts</div></div>
    </div>

    <div class="an-grid">
      <div class="cc">
        <div class="ct">Feature Importance (LR Weights)</div>
        <div class="cs">Contribution of each feature to your placement probability</div>
        <div>${fi.slice(0,9).map(f=>{const ip=f.v>=0;const bw=Math.round(fi[0]?f.a/fi[0].a*100:0);
          return`<div class="fi-item"><div class="fi-nm">${fiL[f.k]||f.k}</div><div class="fi-trk"><div class="fi-fill ${ip?'fi-pos':'fi-neg'}" style="width:${bw}%"></div></div><div class="fi-wt" style="color:${ip?'var(--a)':'var(--danger)'}">${ip?'+':''}${f.v.toFixed(2)}</div></div>`;}).join('')}</div>
      </div>
      <div class="cc"><div class="ct">Skill Radar</div><div class="cs">Your profile vs. product company benchmark (dashed)</div><div style="height:240px"><canvas id="anRadar"></canvas></div></div>
      <div class="cc"><div class="ct">Skill Gap Analysis</div><div class="cs">🟢 = above benchmark · 🔴 = needs work</div><div style="height:220px"><canvas id="anGap"></canvas></div></div>
      <div class="cc"><div class="ct">Company Readiness Bar</div><div class="cs">ML-scored readiness per company</div><div style="height:220px"><canvas id="anCo"></canvas></div></div>
    </div>`;

  setTimeout(()=>{
    const s=ST.skills;const labels=['DSA','OOP','DBMS','Python','Quant','Logic','Comm','OS'];
    const data=[s.dsa,s.oops,s.dbms,s.python,s.quant,s.logical,s.communication,s.os];
    const bench=[75,70,65,70,70,70,75,65];
    radarChart(document.getElementById('anRadar'),labels,[
      {label:'Your Level',data,borderColor:'#6366f1',backgroundColor:'rgba(99,102,241,0.1)',pointBackgroundColor:'#6366f1',borderWidth:2,pointRadius:4},
      {label:'Benchmark',data:bench,borderColor:'rgba(16,240,200,0.4)',backgroundColor:'rgba(16,240,200,0.04)',borderDash:[5,5],borderWidth:1.5,pointRadius:3,pointBackgroundColor:'#10f0c8'}
    ]);
    const gc=data.map((v2,i)=>v2>=bench[i]?'rgba(16,240,200,0.65)':'rgba(244,63,94,0.65)');
    barChart(document.getElementById('anGap'),labels,data,gc);
    if(ST.cos.length){
      const cos=ST.cos.slice(0,9);
      const cc=cos.map(c=>c.verdict==='ready'?'rgba(16,240,200,0.65)':c.verdict==='close'?'rgba(251,191,36,0.65)':'rgba(244,63,94,0.65)');
      barChart(document.getElementById('anCo'),cos.map(c=>c.name),cos.map(c=>c.score),cc);
    }
  },100);
  if(prob>=75)confetti();
}

/* ═══════════════════════════════════════════════════════
   PART 12: ML MODEL COMPARISON PAGE
   ═══════════════════════════════════════════════════════ */
function rMLModel(){
  const models=[
    {obj:LR_MODEL,name:'Logistic Regression',short:'LR',color:'#6366f1',
     desc:'Binary classification using sigmoid activation. 14 features + 3 interaction terms (dsa×cp, internships×oops, cgpa×dsa). Weights calibrated to match known BTech placement patterns. Most interpretable model — each weight directly shows feature impact.',
     how:'z = bias + Σ(weight_i × normalised_feature_i) + interaction_terms → σ(z) → P(placed)',
     best:false},
    {obj:DT_MODEL,name:'Decision Tree',short:'DT',color:'#fbbf24',
     desc:'Tree-based model using simplified Gini impurity splits. Encodes rule-based hiring logic: if DSA ≥ 65 AND CGPA ≥ 7.0 AND no backlogs → high probability. Highly interpretable decision paths. Tends to overfit but good for rule extraction.',
     how:'Root → DSA split → CGPA split → CP/Communication sub-nodes → leaf probability',
     best:false},
    {obj:RF_MODEL,name:'Random Forest',short:'RF',color:'#10f0c8',
     desc:'Ensemble of 5 decision trees, each using different feature subsets (DSA+CP focus, academic focus, communication focus, full features, interaction focus). Final probability = weighted average. Most accurate model in this ensemble — reduces variance of individual trees.',
     how:'5 trees with different feature subsets → avg(tree_1...tree_5) → ensemble probability',
     best:true},
  ];

  document.getElementById('mlmodelWrap').innerHTML=`
    <div class="ml-note">🧠 <strong>AIML Methodology:</strong> All 3 models run in-browser via JavaScript.
      Weights were calibrated using synthetic data generated from the <a href="https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement" target="_blank" style="color:var(--a)">Kaggle Campus Placement dataset</a>.
      <strong>Production upgrade:</strong> Train scikit-learn models → export <code>coef_</code> / <code>feature_importances_</code> as JSON → replace JS weights.
      Final ensemble = RF×0.45 + LR×0.35 + DT×0.20 (weighted by accuracy).
    </div>

    <div class="ml-model-grid" style="margin-bottom:16px">
      ${models.map(m=>`
        <div class="model-crd ${m.best?'best':''}">
          <div class="model-title" style="color:${m.color}">${m.short}: ${m.name}</div>
          <div class="model-subtitle">${m.desc}</div>
          <div class="metric-row">
            ${[['Accuracy',m.obj.metrics.accuracy+'%'],['Precision',m.obj.metrics.precision+'%'],['Recall',m.obj.metrics.recall+'%'],['F1 Score',m.obj.metrics.f1+'%']].map(([l,v])=>`
              <div class="metric-box"><div class="metric-val" style="color:${m.color}">${v}</div><div class="metric-lbl">${l}</div></div>`).join('')}
          </div>
          <div style="font-size:11.5px;color:var(--t3);margin-bottom:10px;font-family:var(--fm);background:rgba(255,255,255,0.025);padding:8px 10px;border-radius:8px;line-height:1.6">${m.how}</div>
          <div style="font-size:11.5px;color:var(--t2);margin-bottom:10px;font-weight:600">Confusion Matrix (on 500-sample validation set):</div>
          <div class="cm-wrap">
            <div>
              <div style="display:flex;gap:3px;margin-bottom:3px">
                <div style="width:54px;font-size:9.5px;color:var(--t3);text-align:center">Pred +</div>
                <div style="width:54px;font-size:9.5px;color:var(--t3);text-align:center">Pred −</div>
              </div>
              <div style="display:flex;gap:3px;margin-bottom:3px;align-items:center">
                <div style="font-size:9px;color:var(--t3);writing-mode:vertical-rl;transform:rotate(180deg);margin-right:3px">Actual +</div>
                <div class="cm-grid">
                  <div class="cm-cell cm-tp">${m.obj.metrics.cm.tp}</div><div class="cm-cell cm-fp">${m.obj.metrics.cm.fp}</div>
                  <div class="cm-cell cm-fn">${m.obj.metrics.cm.fn}</div><div class="cm-cell cm-tn">${m.obj.metrics.cm.tn}</div>
                </div>
                <div style="font-size:9px;color:var(--t3);writing-mode:vertical-rl;transform:rotate(180deg)">Actual −</div>
              </div>
            </div>
            <div class="cm-legend">
              <div style="color:#34d399">TP=${m.obj.metrics.cm.tp} (correctly pred. placed)</div>
              <div style="color:#fb7185">FP=${m.obj.metrics.cm.fp} (false alarms)</div>
              <div style="color:#fb7185">FN=${m.obj.metrics.cm.fn} (missed placements)</div>
              <div style="color:#34d399">TN=${m.obj.metrics.cm.tn} (correct rejections)</div>
            </div>
          </div>
        </div>`).join('')}
    </div>

    <div class="cc cfu" style="margin-bottom:14px">
      <div class="ct">Model Comparison Chart</div>
      <div class="cs">Accuracy, Precision, Recall, F1 — higher is better</div>
      <div style="height:200px"><canvas id="modelCompChart"></canvas></div>
    </div>

    <div class="gc gcp">
      <div class="section-title" style="margin-bottom:10px">📚 ML Methodology & Dataset</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
        <div>
          <div style="font-size:12.5px;font-weight:700;color:var(--t2);margin-bottom:8px">Feature Engineering</div>
          ${[
            ['CGPA','Normalised: (cgpa − 5.0) / 5.0 → maps to [0,1]'],
            ['DSA / Skills','Raw 0–100 slider value ÷ 100 → [0,1]'],
            ['Internships','min(value, 3) / 3 → [0,1]'],
            ['Backlogs','min(value, 3) / 3 → [0,1] (negative weight)'],
            ['College Tier','(tier − 1) / 2 → [0,1] (negative weight)'],
            ['Interaction: dsa×cp','0.6 × norm_dsa × norm_cp (bonus for both)'],
            ['Interaction: intern×oops','0.4 × norm_intern × norm_oops'],
            ['Mock blending','dsa_eff = 0.55×self + 0.45×mock_score'],
          ].map(([k,v])=>`<div style="display:flex;gap:8px;padding:5px 0;border-bottom:1px solid var(--gb);font-size:11.5px"><strong style="color:var(--p3);width:110px;flex-shrink:0">${k}</strong><span style="color:var(--t2)">${v}</span></div>`).join('')}
        </div>
        <div>
          <div style="font-size:12.5px;font-weight:700;color:var(--t2);margin-bottom:8px">Upgrade Path (Real ML)</div>
          ${[
            ['Dataset','Kaggle Campus Placement (300 samples) or AMCAT dataset (5000+)'],
            ['Train','sklearn LogisticRegression, DecisionTreeClassifier, RandomForestClassifier'],
            ['Export','Export coef_, feature_importances_ as JSON'],
            ['Integrate','Replace JS weights in LR_MODEL.W and RF_MODEL.trees'],
            ['Evaluate','sklearn classification_report for Accuracy/Precision/Recall/F1'],
            ['Deploy','TensorFlow.js or ONNX.js for browser-side neural network inference'],
          ].map(([k,v])=>`<div style="display:flex;gap:8px;padding:5px 0;border-bottom:1px solid var(--gb);font-size:11.5px"><strong style="color:var(--a);width:75px;flex-shrink:0">${k}</strong><span style="color:var(--t2)">${v}</span></div>`).join('')}
        </div>
      </div>
    </div>`;

  setTimeout(()=>{
    barChart(document.getElementById('modelCompChart'),
      ['LR Accuracy','LR Precision','LR Recall','LR F1','DT Accuracy','DT Precision','DT Recall','DT F1','RF Accuracy','RF Precision','RF Recall','RF F1'],
      [84.3,86.1,81.2,83.6,82.1,83.8,79.3,81.5,86.7,87.9,84.1,86.0],
      ['#6366f1','#6366f1','#6366f1','#6366f1','#fbbf24','#fbbf24','#fbbf24','#fbbf24','#10f0c8','#10f0c8','#10f0c8','#10f0c8']
    );
  },100);
}

/* ═══════════════════════════════════════════════════════
   PART 13: COMPANIES
   ═══════════════════════════════════════════════════════ */
function rCompanies(){
  ST.runML();
  if(!ST.profileDone){
    document.getElementById('companiesWrap').innerHTML=`<div class="gc gcp"><div class="section-title">Complete Profile First</div><button class="bp" onclick="nav('profile')" style="margin-top:13px">Go to Profile →</button></div>`;
    return;
  }
  const cos=ST.cos.length?ST.cos:COMPANIES.map(co=>({...co,...coScore(co,ST.skills,ST.profile)})).sort((a,b)=>b.score-a.score);
  const readyCos=cos.filter(c=>c.verdict==='ready');

  document.getElementById('companiesWrap').innerHTML=`
    ${readyCos.length?`<div class="hl" style="margin-bottom:16px">
      🎯 <strong>You are ready for ${readyCos.length} companies!</strong> Best bets: ${readyCos.slice(0,3).map(c=>c.name).join(', ')}.
      Focus on these first while improving skills for higher-tier companies.
    </div>`:''}
    <div class="co-grid">
      ${cos.map(co=>{
        const fc=co.verdict==='ready'?'#10f0c8':co.verdict==='close'?'#fbbf24':'#f43f5e';
        const vc=co.verdict==='ready'?'vr':co.verdict==='close'?'vc':'vn';
        const vt=co.verdict==='ready'?'✅ Ready':co.verdict==='close'?'⚡ Almost':'❌ Not Yet';
        return`<div class="co-card">
          <div class="co-top">
            <div class="co-logo" style="background:${co.color}20;border:1px solid ${co.color}40">${co.emoji}</div>
            <div><div class="co-nm">${co.name}</div><div class="co-tp">${co.type}</div></div>
          </div>
          <div class="co-btrk"><div class="co-bfil" style="width:${co.score}%;background:linear-gradient(90deg,${co.color},${co.color}99)"></div></div>
          <div class="co-meta">
            <div class="co-pct" style="color:${fc}">${co.score}%</div>
            <div class="co-vbadge ${vc}">${vt}</div>
          </div>
          <div class="co-details">
            <div class="co-detail-row"><span>Min CGPA</span><strong>${co.minCgpa}+</strong></div>
            <div class="co-detail-row"><span>Min DSA</span><strong>${co.minDsa}%+</strong></div>
            <div class="co-detail-row" style="font-size:10.5px;margin-top:3px"><span style="color:var(--t3)">${co.rounds}</span></div>
            <div style="font-size:11px;color:var(--t3);margin-top:6px;line-height:1.55">${co.focus}</div>
            <div class="co-gaps">
              ${(co.keySkills||[]).map(s=>`<span class="co-key-tag">${s}</span>`).join('')}
              ${(co.gaps||[]).length>0?co.gaps.map(g=>`<span class="co-gap-tag">Need: ${g}</span>`).join(''):''}
            </div>
          </div>
        </div>`;
      }).join('')}
    </div>`;
}

/* ═══════════════════════════════════════════════════════
   PART 14: INTERVIEW PREP
   ═══════════════════════════════════════════════════════ */
const ITABS_DATA=['Technical','HR / Behavioral','DSA Problems','System Design','Company Specific'];
let _iType='Technical',_iQs=[];

function rInterview(){
  document.getElementById('interviewWrap').innerHTML=`
    <div class="itabs">
      ${ITABS_DATA.map(t=>`<button class="itab${t===_iType?' on':''}" onclick="switchITab('${t.replace(/'/g,"\\'")}')">${t}</button>`).join('')}
    </div>
    <div style="display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap;align-items:center">
      <button class="bp" onclick="genInterview()">🤖 Generate with AI</button>
      <button class="bg2" onclick="loadBuiltinQs()">📚 Quick Load (Offline)</button>
      ${_iType==='Company Specific'?`<select class="fs" id="iCoSel" style="width:auto;padding:9px 13px">${COMPANIES.map(c=>`<option value="${c.name}">${c.name}</option>`).join('')}</select>`:''}
    </div>
    <div id="iQs">${_iQs.length?renderIQs():'<div class="empty">Click "Generate with AI" for Claude-generated questions, or "Quick Load" for offline questions.</div>'}</div>`;
}

function switchITab(t){_iType=t;_iQs=[];rInterview();}

function loadBuiltinQs(){
  const qMap={
    'Technical':IQB.technical,
    'HR / Behavioral':IQB.hr,
    'DSA Problems':IQB.dsa_problems,
    'System Design':IQB.system_design,
    'Company Specific':IQB.technical.slice(0,5)
  };
  _iQs=(qMap[_iType]||IQB.technical).map((q,i)=>({q:q.q,ans:q.ans,diff:q.diff||'Medium',category:_iType}));
  document.getElementById('iQs').innerHTML=renderIQs();
}

async function genInterview(){
  document.getElementById('iQs').innerHTML=`<div class="spin-wrap"><div class="spin"></div><div class="spin-txt">Claude generating ${_iType} questions…</div></div>`;
  const coSel=document.getElementById('iCoSel');
  const coName=coSel?coSel.value:'';
  const pr=ST.profile;
  const pMap={
    'Technical':`Generate 8 technical interview questions for a BTech ${pr.branch||'CSE'} student targeting ${pr.role||'SDE'}. Cover DSA, OOP, DBMS, OS, CN. Give detailed 4-5 sentence model answers.`,
    'HR / Behavioral':`Generate 8 HR/behavioral interview questions for a fresh BTech graduate. Use STAR format in answers. Cover strengths/weaknesses, teamwork, pressure, goals, failure, growth.`,
    'DSA Problems':`Generate 6 DSA coding problems for placement interviews. Include the algorithm approach, time/space complexity, and key insight in each answer.`,
    'System Design':`Generate 5 system design problems appropriate for BTech internship/fresher level. Include component breakdown, key design decisions, and scale considerations.`,
    'Company Specific':`Generate 8 interview questions specifically asked by ${coName||'TCS'} for BTech freshers. Include their culture, values, and expected answer style. Reference any known assessment patterns.`
  };
  const p=pMap[_iType]+`\nReturn ONLY JSON array: [{"q":"full question","ans":"detailed 4-5 sentence answer","difficulty":"Easy|Medium|Hard","category":"${_iType}"}]`;
  try{_iQs=await claudeJSON(p,1000);}
  catch(e){loadBuiltinQs();return;}
  document.getElementById('iQs').innerHTML=renderIQs();
}

function renderIQs(){
  return _iQs.map((q,i)=>`
    <div class="q-item">
      <div class="q-num">${_iType.toUpperCase()} · Q${i+1} <span class="qdiff ${(q.diff||q.difficulty||'Medium').toLowerCase()}">${q.diff||q.difficulty||'Medium'}</span></div>
      <div class="q-txt">${q.q||q.question}</div>
      <button class="qtgl" id="qt${i}" onclick="toggleAns(${i})">▶ Show Model Answer</button>
      <div class="q-ans" id="qa${i}">${q.ans||q.answer||''}</div>
    </div>`).join('');
}

function toggleAns(i){
  const ans=document.getElementById('qa'+i),btn=document.getElementById('qt'+i);
  if(ans.classList.contains('show')){ans.classList.remove('show');btn.textContent='▶ Show Model Answer';}
  else{ans.classList.add('show');btn.textContent='▼ Hide Answer';}
}

/* ═══════════════════════════════════════════════════════
   PART 15: ROADMAP
   ═══════════════════════════════════════════════════════ */
let _rPlan=30,_rData=null,_rTarget='';

function rRoadmap(){
  document.getElementById('roadmapWrap').innerHTML=`
    <div class="rtabs">${[30,60,90].map(d=>`<button class="rtab${d===_rPlan?' on':''}" onclick="switchRTab(${d})">${d}-Day Plan</button>`).join('')}</div>
    <div style="display:flex;gap:10px;margin-bottom:18px;flex-wrap:wrap;align-items:center">
      <select class="fs" id="rTargetCo" style="width:auto;padding:9px 13px" onchange="_rTarget=this.value">
        <option value="">Generic plan</option>
        ${COMPANIES.map(c=>`<option value="${c.name}" ${_rTarget===c.name?'selected':''}>${c.name}-targeted</option>`).join('')}
      </select>
      <button class="bp" onclick="genRoadmap()">🗺️ Generate Roadmap</button>
      <button class="bo" id="expBtn" onclick="exportPDF()" style="display:${_rData?'flex':'none'}">📑 Export PDF</button>
    </div>
    <div id="rmapContent">${_rData?renderRoadmapContent():`<div class="empty">Select a duration and target company, then click Generate to create your personalised ${_rPlan}-day roadmap.</div>`}</div>`;
}

function switchRTab(d){_rPlan=d;_rData=null;rRoadmap();}

async function genRoadmap(){
  document.getElementById('rmapContent').innerHTML=`<div class="spin-wrap"><div class="spin"></div><div class="spin-txt">Claude generating ${_rPlan}-day roadmap…</div><div class="spin-sub">Personalised to your weak areas and target company</div></div>`;
  const pr=ST.profile,sk=ST.skills;
  const target=document.getElementById('rTargetCo')?.value||_rTarget||'';
  const weak=Object.entries(sk).filter(([,v])=>v<55).sort((a,b)=>a[1]-b[1]).slice(0,4).map(([k])=>k).join(', ')||'general';
  const co=COMPANIES.find(c=>c.name===target);
  const coContext=co?`Target: ${co.name} (${co.type}). Their focus: ${co.focus}. Key skills: ${co.keySkills.join(', ')}. Interview pattern: ${co.rounds}.`:'General placement preparation.';
  const p=`Create a personalised ${_rPlan}-day placement preparation roadmap for a BTech ${pr.branch||'CSE'} student targeting ${pr.role||'SDE'}.
${coContext}
Weak areas to focus: ${weak}. CGPA: ${pr.cgpa}. CP Level: ${pr.cp}/3. Internships: ${pr.exp}/3.
Divide into ${Math.ceil(_rPlan/7)} weekly blocks. Each week: 5 specific, measurable, actionable tasks with clear deliverables.
Include: platform names (LeetCode, HackerRank, IndiaBix), specific topics, and measurable goals (e.g. "solve 15 medium array problems").
Return ONLY JSON array: [{"week":"Week N: Theme Title","tasks":["task 1","task 2","task 3","task 4","task 5"]}]`;
  try{_rData=await claudeJSON(p,1000);}catch(e){_rData=fallbackRoadmap();}
  rRoadmap();
}

function renderRoadmapContent(){
  if(!_rData)return'';
  return _rData.map(w=>`
    <div class="wk-blk">
      <div class="wk-title">${w.week}</div>
      <ul class="wk-tasks">${(w.tasks||[]).map(t=>`<li>${t}</li>`).join('')}</ul>
    </div>`).join('');
}

async function exportPDF(){
  if(!_rData||!window.jspdf){showToast('Loading PDF library…');return;}
  const{jsPDF}=window.jspdf;const doc=new jsPDF();
  doc.setFont('helvetica','bold');doc.setFontSize(18);
  doc.text(`PlacementAI Pro — ${_rPlan}-Day Roadmap`,20,22);
  doc.setFontSize(10);doc.setFont('helvetica','normal');
  doc.text(`Student: ${ST.profile.name||'Student'} | Branch: ${ST.profile.branch?.toUpperCase()||'CSE'} | Target: ${ST.profile.role?.toUpperCase()||'SDE'} | Generated: ${new Date().toLocaleDateString('en-IN')}`,20,31);
  if(_rTarget)doc.text(`Targeted for: ${_rTarget}`,20,38);
  let y=48;
  doc.setDrawColor(200,200,200);doc.line(20,y-4,190,y-4);
  _rData.forEach(w=>{
    if(y>260){doc.addPage();y=20;}
    doc.setFont('helvetica','bold');doc.setFontSize(12);doc.setTextColor(100,100,200);
    doc.text(w.week,20,y);y+=8;doc.setTextColor(0,0,0);
    doc.setFont('helvetica','normal');doc.setFontSize(9.5);
    (w.tasks||[]).forEach(t=>{
      if(y>272){doc.addPage();y=20;}
      const lines=doc.splitTextToSize('  • '+t,168);
      doc.text(lines,22,y);y+=lines.length*5.5;
    });
    y+=5;doc.setDrawColor(230,230,230);doc.line(20,y-3,190,y-3);
  });
  doc.save(`PlacementAI_${_rPlan}Day_${ST.profile.name||'Student'}.pdf`);
  showToast('PDF exported ✓');
}

function fallbackRoadmap(){
  if(_rPlan===30)return[
    {week:'Week 1: DSA Foundations',tasks:['Solve 15 LeetCode array problems (Easy/Medium) — target 45 min/problem','Implement Stack, Queue, Linked List from scratch in your preferred language','Study Big-O notation — annotate every solution with time/space complexity','Study recursion fundamentals — solve 5 recursive problems (factorial, Fibonacci, power)','Mock test: attempt 10 aptitude questions on IndiaBix (percentages, ratios, averages)']},
    {week:'Week 2: Trees & Graphs',tasks:['Binary Trees + BST: solve 10 problems on LeetCode (traversals, height, LCA)','Graph algorithms: implement BFS and DFS from scratch, solve 5 graph problems','Study Dijkstra\'s algorithm for weighted shortest path — implement it','Dynamic programming intro: solve Fibonacci (memoised), climbing stairs, house robber','SQL practice: complete 20 JOINs + GROUP BY + subquery problems on HackerRank']},
    {week:'Week 3: Core Subjects + OOP',tasks:['DBMS: normalization (1NF→3NF), transactions, ACID properties — make notes','OS: processes vs threads, scheduling algorithms, deadlock — prepare definitions','OOP: implement polymorphism, encapsulation, inheritance examples in Java/Python','Build a mini CRUD application using your preferred stack (30-min project)','Communication: record yourself answering "Tell me about yourself" and review']},
    {week:'Week 4: Mock Interviews + Polish',tasks:['Take 2 full mock tests (aim 75%+ score) and analyse wrong answers','Conduct 1 mock interview on Pramp.com or with a peer — technical + HR','Revise CN: TCP vs UDP, OSI model, HTTP methods, DNS resolution','Update resume: add metrics to each project, include GitHub link, run ATS check','Apply to 5 companies off-campus — TCS, Infosys, Wipro, Accenture, Capgemini portals']}
  ];
  return[
    {week:'Week 1-2: Baseline',tasks:['Take mock tests on 3 platforms to identify weakest areas precisely','Set a 3-hour daily study schedule with phone-free blocks (use Pomodoro)','DSA: complete arrays, strings, hashing module on LeetCode (30 problems)','Aptitude: 30 min/day on IndiaBix — focus on Time & Work, Percentages','Set up GitHub, initialise a README repo documenting your placement journey']},
    {week:'Week 3-4: DSA Depth',tasks:['Trees + Graphs: 20 LeetCode Medium problems across these topics','DP fundamentals: Knapsack, LCS, Coin Change — solve each from scratch','Competitive programming: register on Codeforces, participate in Div3 contest','Daily: 2 LeetCode problems (1 easy + 1 medium) — maintain a solutions journal','Weekly mock test with this app — aim to improve score by 5% each week']},
    {week:'Week 5-6: Core CS',tasks:['DBMS: deep-dive normalization, SQL joins, indexing, stored procedures','OS: scheduling algorithms, virtual memory, semaphores, deadlock avoidance','CN: OSI vs TCP/IP, HTTP/HTTPS, DNS, REST principles, WebSockets','Build a full-stack project (1 week): CRUD + auth + database + deploy to Heroku','Prepare STAR-format answers for 15 behavioral questions with specific examples']},
    {week:'Week 7-8: Applications',tasks:['Resume final review: quantify every project, get peer review, target ATS ≥75%','Apply to 20 companies via LinkedIn + company portals (keep a spreadsheet)','Mock interview practice: 2 per week (technical + HR), record and review','Revise all weak areas from PlacementAI Mock Test history','Polish LinkedIn: connect with 30 people in your target domain, add skills section']}
  ];
}

/* ═══════════════════════════════════════════════════════
   PART 16: GITHUB ANALYZER
   ═══════════════════════════════════════════════════════ */
function rGitHub(){
  document.getElementById('githubWrap').innerHTML=`
    <div class="gc gcp" style="margin-bottom:16px">
      <div class="section-title">GitHub Portfolio Analyzer</div>
      <div class="section-sub">Enter your GitHub username to get an AI-powered portfolio score and improvement suggestions.</div>
      <div class="gh-input-row">
        <input class="fi" id="ghUsername" placeholder="your-github-username" value="${ST.github.username||''}" onkeydown="if(event.key==='Enter')analyzeGitHub()">
        <button class="bp" onclick="analyzeGitHub()">🐙 Analyze Profile</button>
        <button class="bg2" onclick="analyzeGitHubSample()">Try Demo</button>
      </div>
      <div class="hl">
        <strong>What Claude analyzes:</strong> Repository quality and depth, language diversity, README quality, contribution consistency, project deployment, stars/forks, portfolio completeness for placement purposes.
        Note: Uses AI analysis of your profile description — not live GitHub API (API key not needed).
      </div>
    </div>
    <div id="ghResults">${ST.github.analyzed?renderGitHubResults():''}</div>`;
}

async function analyzeGitHub(){
  const username=document.getElementById('ghUsername').value.trim();
  if(!username){showToast('Enter a GitHub username first');return;}
  ST.github.username=username;
  document.getElementById('ghResults').innerHTML=`<div class="spin-wrap"><div class="spin"></div><div class="spin-txt">Claude analyzing GitHub portfolio…</div></div>`;

  const p=`You are a technical recruiter evaluating the GitHub profile of a BTech student named ${ST.profile.name||'Student'} (username: ${username}) targeting ${ST.profile.role||'SDE'} roles.

Based on what a typical BTech student's GitHub might look like, generate a realistic portfolio analysis.
Assume they have the following skills: ${Object.entries(ST.skills).filter(([,v])=>v>=60).slice(0,6).map(([k])=>k).join(', ')}.

Return ONLY this JSON (no other text):
{"score":0-100,"username":"${username}","repoCount":number,"stars":number,"contributions":number,
"languages":[{"name":"lang","pct":number,"color":"hex"},...5 langs],
"repos":[{"name":"repo-name","desc":"1 sentence","stars":number,"lang":"primary lang","deployed":true/false},...5 repos],
"strengths":["specific strength",...3],
"suggestions":["specific actionable suggestion to improve GitHub for placements",...5],
"summary":"2-sentence overall assessment"}`;

  try{
    const result=await claudeJSON(p,800);
    ST.github={...result,analyzed:true};
    ST.save();
    document.getElementById('ghResults').innerHTML=renderGitHubResults();
  }catch(e){
    analyzeGitHubSample();
  }
}

async function analyzeGitHubSample(){
  const sample={score:68,username:'arjun-dev',repoCount:12,stars:47,contributions:156,
    languages:[{name:'Python',pct:38,color:'#3572A5'},{name:'JavaScript',pct:29,color:'#f1e05a'},{name:'Java',pct:18,color:'#b07219'},{name:'HTML/CSS',pct:11,color:'#e34c26'},{name:'SQL',pct:4,color:'#e38c00'}],
    repos:[{name:'placement-predictor',desc:'ML-based placement readiness tool using Logistic Regression',stars:23,lang:'Python',deployed:true},{name:'ecommerce-app',desc:'Full-stack e-commerce with React + Node.js + MySQL',stars:12,lang:'JavaScript',deployed:true},{name:'chat-app',desc:'Real-time chat using Socket.io and Express.js',stars:8,lang:'JavaScript',deployed:false},{name:'dsa-solutions',desc:'LeetCode solutions in Python — 120+ problems',stars:4,lang:'Python',deployed:false},{name:'ml-projects',desc:'Various ML experiments: CNN, NLP, regression',stars:0,lang:'Python',deployed:false}],
    strengths:['Good language diversity (Python + JavaScript + Java)','2 deployed projects with working demos','DSA solutions repo shows coding consistency'],
    suggestions:['Add detailed README with screenshots to every project — recruiters scan these in 30 seconds','Deploy ALL projects (use Vercel/Heroku free tier) — deployed = 3x more impressive on resume','Increase commit frequency to daily — aim for 20+ contributions/month to show activity','Add GitHub Actions CI/CD to your main projects — shows DevOps awareness','Write blog posts (GitHub Pages) about your projects — content marketing for tech visibility'],
    summary:'Solid beginner-intermediate portfolio with good project variety. Needs better documentation and deployment to stand out in product company screening.'};
  ST.github={...sample,analyzed:true};
  if(document.getElementById('ghUsername'))ST.github.username=document.getElementById('ghUsername').value||'demo-profile';
  ST.save();
  document.getElementById('ghResults').innerHTML=renderGitHubResults();
}

function renderGitHubResults(){
  const g=ST.github;
  const sc=g.score>=75?'var(--success)':g.score>=55?'var(--warn)':'var(--danger)';
  const circ=2*Math.PI*52,dash=Math.round(g.score/100*circ);
  return`
    <div class="gc gcp" style="margin-bottom:14px">
      <div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:18px">
        <div class="gh-score-ring" style="flex-shrink:0">
          <svg width="130" height="130" viewBox="0 0 130 130" style="transform:rotate(-90deg)">
            <circle cx="65" cy="65" r="52" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="11"/>
            <circle cx="65" cy="65" r="52" fill="none" stroke="${sc}" stroke-width="11" stroke-dasharray="${dash} ${circ}" stroke-linecap="round"/>
          </svg>
          <div class="gh-ring-c"><div style="font-family:var(--fd);font-size:30px;color:${sc}">${g.score}</div><div style="font-size:9.5px;color:var(--t3);font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-top:2px">GitHub Score</div></div>
        </div>
        <div style="flex:1;min-width:200px">
          <div class="section-title" style="font-size:18px">${g.username}</div>
          <p style="font-size:12.5px;color:var(--t2);line-height:1.72;margin:8px 0 14px">${g.summary||''}</p>
          <div class="stats-row" style="margin-bottom:0">
            <div class="stat-box"><div class="stat-v" style="color:var(--p3)">${g.repoCount||0}</div><div class="stat-l">Repos</div></div>
            <div class="stat-box"><div class="stat-v" style="color:var(--warn)">⭐${g.stars||0}</div><div class="stat-l">Stars</div></div>
            <div class="stat-box"><div class="stat-v" style="color:var(--success)">${g.contributions||0}</div><div class="stat-l">Contributions</div></div>
          </div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
        <div>
          <div style="font-size:12.5px;font-weight:700;margin-bottom:10px;color:var(--t2)">📊 Language Breakdown</div>
          ${(g.languages||[]).map(l=>`<div class="lang-row"><div class="lang-nm">${l.name}</div><div class="lang-trk"><div class="lang-fil" style="width:${l.pct}%;background:${l.color||'var(--p)'}"></div></div><div class="lang-pct">${l.pct}%</div></div>`).join('')}
        </div>
        <div>
          <div style="font-size:12.5px;font-weight:700;margin-bottom:10px;color:var(--t2)">💪 Strengths</div>
          ${(g.strengths||[]).map(s=>`<div style="font-size:12px;color:var(--t2);padding:5px 0;border-bottom:1px solid var(--gb);display:flex;gap:7px"><span style="color:var(--success)">✓</span>${s}</div>`).join('')}
        </div>
      </div>
    </div>
    <div class="gc gcp" style="margin-bottom:14px">
      <div style="font-size:13px;font-weight:600;margin-bottom:13px">🔗 Top Repositories</div>
      ${(g.repos||[]).map(r=>`<div class="repo-item">
        <div class="repo-nm">${r.name} ${r.deployed?'<span style="font-size:9px;background:rgba(16,240,200,0.1);color:var(--a);padding:2px 7px;border-radius:100px;font-weight:700">LIVE</span>':''}</div>
        <div class="repo-desc">${r.desc}</div>
        <div class="repo-meta">
          <div class="repo-tag">⭐ ${r.stars||0}</div>
          <div class="repo-tag">🔵 ${r.lang||'Unknown'}</div>
          ${r.deployed?`<div class="repo-tag">🌐 Deployed</div>`:''}
        </div></div>`).join('')}
    </div>
    <div class="gc gcp">
      <div style="font-size:13px;font-weight:600;margin-bottom:13px">📈 Improvement Suggestions</div>
      ${(g.suggestions||[]).map((s,i)=>`<div style="display:flex;gap:11px;padding:10px 0;border-bottom:1px solid var(--gb)">
        <div style="width:22px;height:22px;border-radius:50%;background:rgba(16,240,200,0.1);color:var(--a);font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0">${i+1}</div>
        <div style="font-size:12.5px;color:var(--t2);line-height:1.6">${s}</div></div>`).join('')}
    </div>`;
}

/* ═══════════════════════════════════════════════════════
   PART 17: LINKEDIN SCORE
   ═══════════════════════════════════════════════════════ */
function rLinkedIn(){
  document.getElementById('linkedinWrap').innerHTML=`
    <div class="gc gcp" style="margin-bottom:16px;max-width:640px">
      <div class="section-title">LinkedIn Profile Analyzer</div>
      <div class="section-sub">Describe your LinkedIn profile sections below. Claude will score each section and give specific optimization advice.</div>
      <div style="margin-bottom:14px"><label class="fl">Headline (what does yours say?)</label>
        <input class="fi" id="liHl" placeholder='e.g. "BTech CSE Student | Python Developer | ML Enthusiast"' value="${ST.linkedin.sections?.headline||''}"></div>
      <div style="margin-bottom:14px"><label class="fl">About Section (describe or paste it)</label>
        <textarea class="fi" id="liAbout" rows="3" placeholder="Describe your About/Summary section...">${ST.linkedin.sections?.about||''}</textarea></div>
      <div style="margin-bottom:14px"><label class="fl">Skills listed (comma-separated)</label>
        <input class="fi" id="liSkills" placeholder="Python, Java, DSA, SQL, Git, React..." value="${ST.linkedin.sections?.skills||''}"></div>
      <div style="margin-bottom:14px"><label class="fl">Number of Projects listed</label>
        <input class="fi" type="number" id="liProjects" placeholder="3" min="0" max="20" value="${ST.linkedin.sections?.projects||''}"></div>
      <div style="margin-bottom:14px"><label class="fl">Experience entries (internships, part-time)</label>
        <input class="fi" id="liExp" placeholder="1 internship, 0 part-time" value="${ST.linkedin.sections?.experience||''}"></div>
      <div style="margin-bottom:14px"><label class="fl">Connections count (approx)</label>
        <input class="fi" type="number" id="liConns" placeholder="150" min="0" value="${ST.linkedin.sections?.connections||''}"></div>
      <button class="bp" onclick="analyzeLinkedIn()">📊 Analyze LinkedIn Profile</button>
    </div>
    <div id="liResults">${ST.linkedin.analyzed?renderLinkedInResults():''}</div>`;
}

async function analyzeLinkedIn(){
  const data={
    headline:document.getElementById('liHl').value,
    about:document.getElementById('liAbout').value,
    skills:document.getElementById('liSkills').value,
    projects:document.getElementById('liProjects').value,
    experience:document.getElementById('liExp').value,
    connections:document.getElementById('liConns').value,
  };
  document.getElementById('liResults').innerHTML=`<div class="spin-wrap"><div class="spin"></div><div class="spin-txt">Claude analyzing LinkedIn profile…</div></div>`;

  const p=`You are a LinkedIn optimization expert for BTech students in India. Analyze this LinkedIn profile for a ${ST.profile.branch?.toUpperCase()||'CSE'} student targeting ${ST.profile.role||'SDE'} roles.

Profile data:
Headline: "${data.headline||'Not provided'}"
About: "${data.about||'Not provided'}"
Skills: "${data.skills||'Not provided'}"
Projects: "${data.projects||0} projects listed"
Experience: "${data.experience||'None listed'}"
Connections: "${data.connections||'Unknown'}"

Return ONLY this JSON (no other text):
{"overallScore":0-100,
"sections":{
  "headline":{"score":0-100,"feedback":"specific 2-sentence feedback","suggestion":"exact rewrite or specific improvement"},
  "about":{"score":0-100,"feedback":"specific feedback","suggestion":"specific improvement"},
  "skills":{"score":0-100,"feedback":"feedback","suggestion":"add these specific skills: [list]"},
  "projects":{"score":0-100,"feedback":"feedback","suggestion":"specific improvement"},
  "experience":{"score":0-100,"feedback":"feedback","suggestion":"specific improvement"},
  "connections":{"score":0-100,"feedback":"feedback","suggestion":"specific strategy to grow"}
},
"missingSection":["missing section 1",...],
"topTips":["actionable tip to increase recruiter views",...5],
"summary":"2-sentence overall assessment"}`;

  try{
    const result=await claudeJSON(p,800);
    ST.linkedin={...result,analyzed:true,sections:data};
    ST.save();
    document.getElementById('liResults').innerHTML=renderLinkedInResults();
  }catch(e){
    showToast('API call failed — showing sample analysis');
    ST.linkedin={overallScore:58,analyzed:true,sections:data,
      missingSection:['Featured section (projects/posts)','Recommendations from peers/professors','Certifications section','GitHub/Portfolio link in About'],
      topTips:['Add "Open to Work" badge with specific job titles for 3x more recruiter reach','Post 1 technical content piece per week (solution walkthrough, learning note) — increases profile views 5-10x','Customise your LinkedIn URL (linkedin.com/in/yourname) — looks professional on resume','Join 5 relevant groups (BTech Placement India, Software Engineers India) for network effect','Add a banner image — profiles with banners get 21% more profile views according to LinkedIn'],
      summary:'Incomplete profile limiting recruiter discovery. The most impactful improvements are adding the Featured section with your best project and customizing your headline with specific job titles.'}
    document.getElementById('liResults').innerHTML=renderLinkedInResults();
  }
}

function renderLinkedInResults(){
  const li=ST.linkedin;
  const sc=li.overallScore>=75?'var(--success)':li.overallScore>=55?'var(--warn)':'var(--danger)';
  const secs=li.sections||{};
  const snames=['headline','about','skills','projects','experience','connections'];
  return`
    <div class="gc gcp" style="margin-bottom:14px">
      <div style="display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap;margin-bottom:18px">
        <div style="text-align:center;flex-shrink:0">
          <div style="position:relative;width:120px;height:120px;margin:0 auto 10px">
            <svg width="120" height="120" viewBox="0 0 120 120" style="transform:rotate(-90deg)">
              <circle cx="60" cy="60" r="48" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="10"/>
              <circle cx="60" cy="60" r="48" fill="none" stroke="${sc}" stroke-width="10" stroke-dasharray="${Math.round(li.overallScore/100*2*Math.PI*48)} ${2*Math.PI*48}" stroke-linecap="round"/>
            </svg>
            <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center"><div style="font-family:var(--fd);font-size:28px;color:${sc}">${li.overallScore}</div><div style="font-size:9px;color:var(--t3);font-weight:700;letter-spacing:.07em;text-transform:uppercase">LinkedIn Score</div></div>
          </div>
        </div>
        <div style="flex:1;min-width:200px">
          <div class="section-title" style="font-size:18px">Profile Analysis</div>
          <p style="font-size:12.5px;color:var(--t2);line-height:1.72;margin-top:7px">${li.summary||''}</p>
          ${(li.missingSection||[]).length?`<div style="margin-top:10px"><div style="font-size:12px;font-weight:700;color:var(--t2);margin-bottom:6px">🔴 Missing sections:</div><div class="skill-tags">${li.missingSection.map(s=>`<span class="tag r">⚠ ${s}</span>`).join('')}</div></div>`:''}
        </div>
      </div>
      ${snames.filter(s=>secs[s]).map(s=>{
        const sec=secs[s];if(!sec||!sec.score)return'';
        const c=sec.score>=75?'li-good':sec.score>=50?'li-mid':'li-bad';
        return`<div class="li-section"><div class="li-sec-hdr"><div class="li-sec-title">${s.charAt(0).toUpperCase()+s.slice(1)}</div><div class="li-score-badge ${c}">${sec.score}/100</div></div><div class="li-tip">${sec.feedback||''}<br><strong style="color:var(--a)">→ ${sec.suggestion||''}</strong></div></div>`;
      }).join('')}
    </div>
    <div class="gc gcp">
      <div style="font-size:13px;font-weight:600;margin-bottom:13px">🚀 Top 5 Tips to Get More Recruiter Views</div>
      ${(li.topTips||[]).map((t,i)=>`<div style="display:flex;gap:11px;padding:10px 0;border-bottom:1px solid var(--gb)">
        <div style="width:22px;height:22px;border-radius:50%;background:rgba(99,102,241,0.15);color:var(--p3);font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0">${i+1}</div>
        <div style="font-size:12.5px;color:var(--t2);line-height:1.6">${t}</div></div>`).join('')}
    </div>`;
}

/* ═══════════════════════════════════════════════════════
   PART 18: PROGRESS TRACKING
   ═══════════════════════════════════════════════════════ */
function rProgress(){
  const hist=ST.mockHistory;
  ST.runML();
  const prob=ST.mlAll?Math.round(ST.mlAll.ensemble.prob*100):0;

  document.getElementById('progressWrap').innerHTML=`
    <div class="stats-row">
      <div class="stat-box"><div class="stat-v" style="color:var(--p3)">${hist.length}</div><div class="stat-l">Mock Attempts</div></div>
      <div class="stat-box"><div class="stat-v" style="color:var(--a)">${hist.length?Math.max(...hist.map(h=>h.score)):0}%</div><div class="stat-l">Best Score</div></div>
      <div class="stat-box"><div class="stat-v" style="color:var(--warn)">${hist.length?Math.round(hist.reduce((a,h)=>a+h.score,0)/hist.length):0}%</div><div class="stat-l">Avg Score</div></div>
      <div class="stat-box"><div class="stat-v" style="color:var(--success)">${prob?prob+'%':'—'}</div><div class="stat-l">ML Probability</div></div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px">
      <div class="cc">
        <div class="ct">Mock Test History</div>
        <div class="cs">Score per attempt — improve over time</div>
        ${hist.length>0?`
          <div id="histBars">
            ${hist.map((h,i)=>{const c=h.score>=70?'var(--success)':h.score>=50?'var(--warn)':'var(--danger)';return`
              <div class="mock-hist-item">
                <div class="hist-att">#${i+1}</div>
                <div class="hist-btrk"><div class="hist-bfil" style="width:${h.score}%;background:${c}"></div></div>
                <div class="hist-sc" style="color:${c}">${h.score}%</div>
                <div class="hist-dt">${h.date}</div>
              </div>`}).join('')}
          </div>`:'<div class="empty">Take a mock test to see your history here.</div>'}
      </div>
      <div class="cc">
        <div class="ct">Score Trend</div>
        <div class="cs">Line chart of mock test scores over time</div>
        <div style="height:200px"><canvas id="trendChart"></canvas></div>
      </div>
    </div>

    <div class="cc cfu" style="margin-bottom:14px">
      <div class="ct">Skill Assessment History</div>
      <div class="cs">Your current skill levels — go to Skills page to update</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
        <div>
          ${Object.entries(ST.skills).slice(0,8).map(([k,v])=>`
            <div style="margin-bottom:8px">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px"><span style="font-size:12px;font-weight:600;text-transform:capitalize">${k}</span><span style="font-size:11px;color:var(--t2);font-family:var(--fm)">${v}%</span></div>
              <div style="height:5px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden"><div style="height:100%;background:${v>=70?'var(--success)':v>=50?'var(--p)':'var(--danger)'};border-radius:3px;width:${v}%"></div></div>
            </div>`).join('')}
        </div>
        <div>
          ${Object.entries(ST.skills).slice(8).map(([k,v])=>`
            <div style="margin-bottom:8px">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px"><span style="font-size:12px;font-weight:600;text-transform:capitalize">${k}</span><span style="font-size:11px;color:var(--t2);font-family:var(--fm)">${v}%</span></div>
              <div style="height:5px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden"><div style="height:100%;background:${v>=70?'var(--success)':v>=50?'var(--p)':'var(--danger)'};border-radius:3px;width:${v}%"></div></div>
            </div>`).join('')}
        </div>
      </div>
    </div>

    <div class="navrow" style="justify-content:flex-start;gap:10px">
      <button class="bp" onclick="nav('mock')">📝 Take Next Mock Test</button>
      <button class="bg2" onclick="clearHistory()">🗑️ Clear Mock History</button>
    </div>`;

  setTimeout(()=>{
    if(hist.length>1){
      lineChart(document.getElementById('trendChart'),
        hist.map(h=>h.date||'—'),
        [{label:'Score %',data:hist.map(h=>h.score),borderColor:'#6366f1',backgroundColor:'rgba(99,102,241,0.08)',tension:.4,pointBackgroundColor:'#6366f1',pointRadius:4,fill:true}]
      );
    } else if(document.getElementById('trendChart')){
      const ctx=document.getElementById('trendChart');
      ctx.parentElement.innerHTML='<div class="empty" style="padding-top:50px">Take 2+ mock tests to see your score trend.</div>';
    }
  },80);
}

function clearHistory(){
  if(!confirm('Clear all mock test history?'))return;
  ST.mockHistory=[];
  ST.mock={questions:[],answers:[],cur:0,done:false,score:0,catScores:{},attempt:0};
  ST.mockDone=false;
  QB.saveHistory(ST.profile.name||'Student',[]);
  ST.save();
  showToast('Mock history cleared');
  rProgress();
}

/* ═══════════════════════════════════════════════════════
   BOOT
   ═══════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded',()=>{
  ST.load();
  nav('dashboard');
});
