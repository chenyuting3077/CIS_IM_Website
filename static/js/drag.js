
$(document).ready(function() {
  //selecting all required elements
  const dropArea = document.querySelector(".drag-area"),
  dragText = dropArea.querySelector("header"),
  button = dropArea.querySelector("button"),
  input = dropArea.querySelector("input");
  let file; //this is a global variable and we'll use it inside multiple functions

  button.onclick = ()=>{
    input.click(); //if user click on the button then the input also clicked
  }

  input.addEventListener("change", function(){
    //getting user select file and [0] this means if user select multiple files then we'll select only the first one
    file = this.files[0];
    dropArea.classList.add("active");
    // showFile(); //calling function
  });


  //If user Drag File Over DropArea
  dropArea.addEventListener("dragover", (event)=>{
    event.preventDefault(); //preventing from default behaviour
    dropArea.classList.add("active");
  });

  //If user leave dragged File from DropArea
  dropArea.addEventListener("dragleave", ()=>{
    dropArea.classList.remove("active");
  });

  //If user drop File on DropArea
  dropArea.addEventListener("drop", (event)=>{
    event.preventDefault(); 
    file = event.dataTransfer.files[0];
  });
  

  $('#upload-input').change(event =>{
    $('.icon').attr('display','')
    if(event.target.files){
      let filesAmount = event.target.files.length;
      $('#preview_a').html("");
      $('#preview_small').html("");
      function handle(i) {
        if (i >= filesAmount) return;
        // create loader
        let reader = new FileReader();
        // add big picture
        reader.onload = function(event){
          let display= (i!=0)?"style=\"display:none\"":"style=\"display:block\"";
          html = "<img class=\"mySlides\"  src = "+ event.target.result+" "+display+"> </div>";
          $("#preview_a").append(html);
        }
        reader.readAsDataURL(event.target.files[i]);
  
        // add small picture
        let reader_small = new FileReader();
        
        // add big picture
        reader_small.onload = function(event){
          let display= (i!=0)?"":"opacity-off";
          let html = "<div class=\"child\"><img class=\"demo opacity hover-opacity-off "+display+" child-img\" src = "+ event.target.result+" alt=\"image\" onclick=\"currentDiv("+(i+1)+")\"></div>";
          // dict[i] = html;
          $("#preview_small").append(html);
          handle(i+1);
        }
        reader_small.readAsDataURL(event.target.files[i]);
      };
  
      handle(0);
    };
  
    })

    $('#drop_input').change(event =>{
      $('.icon').attr('display','')
      if(event.target.files){
        let filesAmount = event.target.files.length;
        $('#preview_a').html("");
        $('#preview_small').html("");
        function handle(i) {
          if (i >= filesAmount) return;
          // create loader
          let reader = new FileReader();
          // add big picture
          reader.onload = function(event){
            let display= (i!=0)?"style=\"display:none\"":"style=\"display:block\"";
            html = "<img class=\"mySlides\"  src = "+ event.target.result+" "+display+"> </div>";
            $("#preview_a").append(html);
          }
          reader.readAsDataURL(event.target.files[i]);

          // add small picture
          let reader_small = new FileReader();
          
          // add big picture
          reader_small.onload = function(event){
            let display= (i!=0)?"":"opacity-off";
            let html = "<div class=\"child\"><img class=\"demo opacity hover-opacity-off "+display+" child-img\" src = "+ event.target.result+" alt=\"image\" onclick=\"currentDiv("+(i+1)+")\"></div>";
            // dict[i] = html;
            $("#preview_small").append(html);
            handle(i+1);
          }
          reader_small.readAsDataURL(event.target.files[i]);
        };

    handle(0);
  };

  })

});

function scrolll() {
  var left = document.querySelector(".scroll-images");
  left.scrollBy(350, 0)
}

function scrollr() {
  var right = document.querySelector(".scroll-images");
  right.scrollBy(-350, 0)
}

function currentDiv(n) {
  showDivs(slideIndex = n);
}

function showDivs(n) {
  var i;
  var x = document.getElementsByClassName("mySlides");
  var dots = document.getElementsByClassName("demo");
  if (n > x.length) {slideIndex = 1}
  if (n < 1) {slideIndex = x.length}
  for (i = 0; i < x.length; i++) {
    x[i].style.display = "none";
  }
  for (i = 0; i < dots.length; i++) {
    dots[i].className = dots[i].className.replace(" opacity-off", "");
  }
  x[slideIndex-1].style.display = "block";
  dots[slideIndex-1].className += " opacity-off";

}


  