var CMD_CONFIG;
var FAVS;
var news;
var redditing = false, currentsub = "";
var currentArticle = -1;

function more() {
	if(redditing) loadSubReddit(currentsub, 100);
    else
    loadNews(100);
}

function readArticle(index) {
   currentArticle = index;
	
   	var swalbuttons = {
      read: "Articolo completo",
      cancel: "Chiudi",
    };
    
	if(redditing) {
		swalbuttons = {
         read: "Articolo completo",
         rcomments : "Commenti",
         cancel: "Chiudi"
       };    
    }

  swal({
    title: news[index].title,
    text: news[index].description,
    icon: (news[index].urlToImage).replace("&amp;", "&"),

    animation: false,
     buttons: swalbuttons,
  }).then((result) => {
    switch(result) {
    	case "read":
     		var win = window.open(news[index].url, '_blank');
            break;
        case "rcomments":
        	var win = window.open("https://reddit.com" + news[index].permalink, '_blank');
            break;
        default:
      		currentArticle = -1;
    }
  })
}

function loadArticlesFromUrl(url) {
  $.get(url, function(data) {
    var list = $("#feed");
    list.empty();
    news = (redditing)?data.data.children:data.articles;

    for(var i=0; i<news.length;i++)
    {
      var imgUrl;
      if(redditing) {
      	news[i] = news[i].data;
        news[i].source = {};
        news[i].source.name = (news[i].domain.startsWith("self."))? "r": news[i].domain;
      	news[i].urlToImage = (news[i].preview == undefined)?null:news[i].preview.images[0].source.url;
      }
      
      imgUrl = (news[i].urlToImage == null)?"-webkit-linear-gradient(top, #fd0b58 0px, #a32b68 100%)":news[i].urlToImage;
      list.append("<li onclick='readArticle("+ i + ")' class='deck' data-index='" + i + "' > <div class='card card-news'><div class='list-header'> <img class='list-img' src='"+ imgUrl + "'></img> <div class='list-category'>"+news[i].source.name +"</div> <div class='list-title'>" + news[i].title + "</div><div class='list-author'>"+ news[i].author +"</div> <div class='list-datetime'> <i class='fas fa-clock'></i> " + news[i].publishedAt + "</div><div class='list-content'>"+ news[i].description + "</div> <div class='list-footer'> <i class='far fa-thumbs-up likebtn'></i> | <i class='far fa-thumbs-down dislikebtn'></i></div></div></li>");
    }
  });
}

function redditArticle (i) {
	var win = window.open(news[i].url, '_blank');
}

function loadSubReddit(sub, pageSize = 30) {
	var url = 'https://www.reddit.com/r/' + (sub) + "/.json?count=" + pageSize + "&limit=" + pageSize;
  	redditing = true;
    currentsub = sub;
    loadArticlesFromUrl(url);
}

function loadNews(pageSize = 30) {
  //NEWS API: 3091d29cdba743aa947808a266498f36
  var url = 'https://newsapi.org/v2/top-headlines?sources=google-news-it,la-repubblica,ansa,il-sole-24-ore,bbc-news,the-new-york-times,techcrunch,polygon,the-next-web,the-verge,google-news&apiKey=3091d29cdba743aa947808a266498f36&sortBysAvailable=latest&pageSize='+pageSize;
  redditing = false;
  loadArticlesFromUrl(url);
}

function loadItalianNews(pageSize = 30) {
  //NEWS API: 3091d29cdba743aa947808a266498f36
  var url = 'https://newsapi.org/v2/top-headlines?country=it&apiKey=3091d29cdba743aa947808a266498f36&sortBysAvailable=latest&pageSize='+pageSize;
  redditing = false;
  loadArticlesFromUrl(url);
}

function openAllFavs() {
  for(var i=0; i<FAVS.list.length; i++)
  {
    window.open(FAVS.list[i].url, '_blank');
  }
}

function loadConfig() {
  $.getJSON('../config/cmd.json', function(data) {
    CMD_CONFIG = data;
  });
  $.getJSON('../config/favorites.json', function(data) {
    FAVS = data;
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
    var cmdDetected = query.charAt(0)=="!";


    //QUERY CMDs
    if(cmdDetected)
    {
      query = query.substring(1);
      query = query.split(" ");
      cmd = getCmdAction(CMD_CONFIG, query[0]);
      if(cmd == 0) cmd =-1; //CMD NOT FOUND
      query.shift();
      query = query.join(" ");
    }
    
    switch(cmd)
    {
      case -1:
        break;
      case 0:
        var win = window.open("http://www.google.com/search?q="+encodeURIComponent(query), '_blank');
 	  	win.focus();
        break;

      //SPECIAL CMDs
      case 1:
        $("#searchbar").val('');
        redditing = false;
        loadNews(25);
        break;

      case 2:
        $("#searchbar").val('');
        openAllFavs();
        break;

      case 3:
        $("#searchbar").val('');
        man();
        break;
      
      case 4:
        $("#searchbar").val('');
        more();
        break;

	  case "https://www.reddit.com/r/":
        loadSubReddit(query);
        break;
       
      case "italian":
      	loadItalianNews(30);
        break;

      default:
      	redditing = false;
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
          if(currentArticle>=0 && e.keyCode == 110) //right
        	readArticle(currentArticle+1);
        else if(currentArticle>=0 && e.keyCode == 112) //left
        	readArticle(currentArticle-1);
});
