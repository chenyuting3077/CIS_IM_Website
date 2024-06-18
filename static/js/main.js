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
  