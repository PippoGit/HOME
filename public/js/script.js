var CMD_CONFIG;
var news;

function more() {
  loadNews(100);
}

function readArticle(index) {
  var win = window.open(news[index].url, '_blank');
}

function loadArticlesFromUrl(url) {
  $.get(url, function(data) {
    var list = $("#feed");
    list.empty();
    news = data.articles;

    for(var i=0; i<news.length;i++)
    {
      var imgUrl;      
      imgUrl = (news[i].urlToImage == null)?"-webkit-linear-gradient(top, #fd0b58 0px, #a32b68 100%)":news[i].urlToImage;
      list.append("<li onclick='readArticle("+ i + ")' class='deck' data-index='" + i + "' > <div class='card card-news'><div class='list-header'> <img class='list-img' src='"+ imgUrl + "'></img> <div class='list-category'>"+news[i].source.name +"</div> <div class='list-title'>" + news[i].title + "</div><div class='list-author'>"+ news[i].author +"</div> <div class='list-datetime'> <i class='fas fa-clock'></i> " + news[i].publishedAt + "</div><div class='list-content'>"+ news[i].description + "</div> <div class='list-footer'> <i class='far fa-thumbs-up likebtn'></i> | <i class='far fa-thumbs-down dislikebtn'></i></div></div></li>");
    }
  });
}

function loadNews(pageSize = 30) {
  //NEWS API: 3091d29cdba743aa947808a266498f36
  var url = 'https://newsapi.org/v2/top-headlines?sources=google-news-it,la-repubblica,ansa,il-sole-24-ore,bbc-news,the-new-york-times,techcrunch,polygon,the-next-web,the-verge,google-news&apiKey=3091d29cdba743aa947808a266498f36&sortBysAvailable=latest&pageSize='+pageSize;
  loadArticlesFromUrl(url);
}

function loadConfig() {
  $.getJSON('../config/cmd.json', function(data) {
    CMD_CONFIG = data;
  });
}

function getCmdAction(config, val) {
  for(var i=0; i<config.cmds.length; i++)
  {
    if(config.cmds[i].cmd == val)
      return config.cmds[i].url;
  }
  return 0;
}

function man()
{
  $("#searchbar").val('');
  var text = "";// = "<ul>";
  for(var i =0; i < CMD_CONFIG.cmds.length; i++)
  {
    text += " - " + CMD_CONFIG.cmds[i].name + ": !" + CMD_CONFIG.cmds[i].cmd+"\n";
  }
  swal({
    title: "Lista comandi",
    text: text,
    cancel: "Got it"
  });
}

$(document).ready(function() {

  loadConfig();
  loadNews(30);
  $("#searchbar").focus();
  
  $("#search").on('submit', function (e) {
    e.preventDefault();
    var query = $("#searchbar").val();
    var cmd = 0;

    query = query.split(" ");
    cmd = getCmdAction(CMD_CONFIG, query[0]);
    if(cmd == 0) cmd =-1; //CMD NOT FOUND
    query.shift();
    query = query.join(" ");
    
    switch(cmd)
    {
      case 1:
        $("#searchbar").val('');
        loadNews(30);
        break;

      case 3:
        $("#searchbar").val('');
        man();
        break;
      
      case 4:
        $("#searchbar").val('');
        more();
        break;
       
      default:
        var win = window.open(cmd.replace("*", encodeURIComponent(query)), '_blank');
 	  	win.focus();
    }
  });
});

$(document).on('keypress', function (e) {
  if(!$("#searchbar").is(":focus") &&
     ![13,38,40,37,39].includes(e.keyCode)) {
    $("#searchbar").val('');
    $("#searchbar").focus();
  }
});
