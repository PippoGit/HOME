var CONFIG;
var news;

function more() {
  loadNews(100);
}

function readArticle(index) {
  var win = window.open(news[index].link, '_blank');
  win.focus();
}

function loadArticlesFromUrl(url, pageSize = 30) {
  $.get(url, function(data) {
    var list = $("#feed");
    list.empty();
    news = data;

    for(var i=0; i<news.length;i++)
    {
      var imgUrl;      
      imgUrl = (news[i].img == "")?"/img/unipi.gif":news[i].img;
      list.append("<li onclick='readArticle("+ i + ")' class='deck' data-index='" + i + "' > <div class='card card-news'><div class='list-header'> <img class='list-img' src='"+ imgUrl + "'></img> <div class='list-category'>"+news[i].source +"</div> <div class='list-title'>" + news[i].title + "</div><div class='list-author'>"+ news[i].author +"</div> <div class='list-datetime'> <i class='fas fa-clock'></i> " + news[i].datetime + "</div><div class='list-content'>"+ news[i].description + "</div> <div class='list-footer'> <i class='far fa-thumbs-up likebtn'></i> | <i class='far fa-thumbs-down dislikebtn'></i></div></div></li>");
    }
  });
}

function loadNews(pageSize = 30) {
  var url = "http://" + CONFIG.server + ":" + CONFIG.port + CONFIG.API_NEWS_URL + pageSize;
  loadArticlesFromUrl(url, pageSize);
}

function loadConfig() {
  $.getJSON('/config/config.json', function(data) {
    CONFIG = data;
    loadNews(30)
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
  for(var i =0; i < CONFIG.cmds.length; i++)
  {
    text += " - " + CONFIG.cmds[i].name + ": !" + CONFIG.cmds[i].cmd+"\n";
  }
  swal({
    title: "Lista comandi",
    text: text,
    cancel: "Got it"
  });
}

$(document).ready(function() {

  loadConfig();
  $("#searchbar").focus();
  
  $("#search").on('submit', function (e) {
    e.preventDefault();
    var query = $("#searchbar").val();
    var cmd = 0;

    query = query.split(" ");
    cmd = getCmdAction(CONFIG, query[0]);
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
