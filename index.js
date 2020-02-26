CLASSES = {
  0: 'buddha statue',
  1: 'church',
  2: 'moai stone statue',
};

const MODEL_PATH =
    'model.json';

const IMAGE_SIZE = 150;
const TOPK_PREDICTIONS = 3;

let my_model;
const demo = async () => {
  status('Loading model...');

  my_model = await tf.loadLayersModel(MODEL_PATH);


  my_model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  status('');


  const catElement = document.getElementById('church');
  if (catElement.complete && catElement.naturalHeight !== 0) {
    catElement.width = IMAGE_SIZE
    catElement.height = IMAGE_SIZE
    predict(catElement);
    catElement.style.display = '';
  } else {
    catElement.onload = () => {
      catElement.width = IMAGE_SIZE
      catElement.height = IMAGE_SIZE
      predict(catElement);
      catElement.style.display = '';
    }
  }

    const catElement1 = document.getElementById('buddha');
  if (catElement1.complete && catElement1.naturalHeight !== 0) {
    catElement1.width = IMAGE_SIZE
    catElement1.height = IMAGE_SIZE
    predict(catElement1);
    catElement1.style.display = '';
  } else {
    catElement1.onload = () => {
      catElement1.width = IMAGE_SIZE
      catElement1.height = IMAGE_SIZE
      predict(catElement1);
      catElement1.style.display = '';
    }
  }

   const catElement2 = document.getElementById('church2');
  if (catElement2.complete && catElement2.naturalHeight !== 0) {
    catElement2.width = IMAGE_SIZE
    catElement2.height = IMAGE_SIZE
    predict(catElement2);
    catElement2.style.display = '';
  } else {
    catElement2.onload = () => {
      catElement2.width = IMAGE_SIZE
      catElement2.height = IMAGE_SIZE
      predict(catElement2);
      catElement2.style.display = '';
    }
  }


   const catElement3 = document.getElementById('moai');
  if (catElement3.complete && catElement3.naturalHeight !== 0) {
    catElement3.width = IMAGE_SIZE
    catElement3.height = IMAGE_SIZE
    predict(catElement3);
    catElement3.style.display = '';
  } else {
    catElement3.onload = () => {
      catElement3.width = IMAGE_SIZE
      catElement3.height = IMAGE_SIZE
      predict(catElement3);
      catElement3.style.display = '';
    }
  }


   const catElement4 = document.getElementById('buddha2');
  if (catElement4.complete && catElement4.naturalHeight !== 0) {
    catElement4.width = IMAGE_SIZE
    catElement4.height = IMAGE_SIZE
    predict(catElement4);
    catElement4.style.display = '';
  } else {
    catElement4.onload = () => {
      catElement4.width = IMAGE_SIZE
      catElement4.height = IMAGE_SIZE
      predict(catElement);
      catElement4.style.display = '';
    }
  }


   const catElement5 = document.getElementById('moai2');
  if (catElement5.complete && catElement5.naturalHeight !== 0) {
    catElement5.width = IMAGE_SIZE
    catElement5.height = IMAGE_SIZE
    predict(catElement5);
    catElement5.style.display = '';
  } else {
    catElement5.onload = () => {
      catElement5.width = IMAGE_SIZE
      catElement5.height = IMAGE_SIZE
      predict(catElement5);
      catElement5.style.display = '';
    }
  }


   const catElement6 = document.getElementById('church3');
  if (catElement6.complete && catElement6.naturalHeight !== 0) {
    catElement6.width = IMAGE_SIZE
    catElement6.height = IMAGE_SIZE
    predict(catElement6);
    catElement6.style.display = '';
  } else {
    catElement6.onload = () => {
      catElement6.width = IMAGE_SIZE
      catElement6.height = IMAGE_SIZE
      predict(catElement6);
      catElement6.style.display = '';
    }
  }



  document.getElementById('file-container').style.display = '';
};


async function predict(imgElement) {
  status('Predicting...');

  const startTime1 = performance.now();

  let startTime2;
  const logits = tf.tidy(() => {

    const img = tf.browser.fromPixels(imgElement).toFloat();


    const normalized = img.div(255.0);


    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    startTime2 = performance.now();

    return my_model.predict(batched);
  });


  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  status(`Done in ${Math.floor(totalTime1)} ms ` +
      `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);


  showResults(imgElement, classes);
}


async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: CLASSES[topkIndices[i]],
      probability: topkValues[i]
    })
  }
  return topClassesAndProbs;
}



function showResults(imgElement, classes) {
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);

  const probsContainer = document.createElement('div');
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    const classElement = document.createElement('div');
    classElement.className = 'cell';
    classElement.innerText = classes[i].className;
    row.appendChild(classElement);

    const probsElement = document.createElement('div');
    probsElement.className = 'cell';
    probsElement.innerText = classes[i].probability.toFixed(3);
    row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }
  predictionContainer.appendChild(probsContainer);

  predictionsElement.insertBefore(
      predictionContainer, predictionsElement.firstChild);
}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;

    reader.onload = e => {

      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };


    reader.readAsDataURL(f);
  }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');

demo();