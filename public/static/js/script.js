var CONFIG;
var news;

function more() {
  loadNews(100);
}

function like(index) {
  var li = $(event.srcElement).closest('li');

  if(li.hasClass('liked')) {
    li.removeClass('liked');
  }
  else {
    li.removeClass('disliked');
    li.addClass('liked');
  }
}

function dislike(index) {
  var li = $(event.srcElement).closest('li');

  if(li.hasClass('disliked')) {
    li.removeClass('disliked');
  }
  else {
    li.removeClass('liked');
    li.addClass('disliked');
  }
}

function readArticle(index) {
  var win = window.open(news[index].link, '_blank');
  win.focus();
  return false;
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
      list.append("<li data-index='" + i + "' > <div class='card card-news'><div class='list-header'> <img class='list-img' src='"+ imgUrl + "'></img> <div class='list-category'>"+news[i].source +"</div> <a class='list-title' href='#' onclick='readArticle("+ i + ")' >" + news[i].title + "</a><div class='list-author'>"+ news[i].author +"</div> <div class='list-datetime'> <i class='fas fa-clock'></i> " + news[i].datetime + "</div><div class='list-content'>"+ news[i].description + "</div> <div class='list-footer'> <i  onclick='like("+ i + ")'  class='far fa-thumbs-up likebtn'></i> | <i  onclick='dislike("+ i + ")'  class='far fa-thumbs-down dislikebtn'></i></div></div></li>");
    }
  });
}

function loadNews(pageSize = 30) {
  var url = "http://" + CONFIG.server + ":" + CONFIG.port + CONFIG.API_FEED_URL + pageSize;
  loadArticlesFromUrl(url, pageSize);
}

function learning(pageSize = 50) {
  var url = "http://" + CONFIG.server + ":" + CONFIG.port + CONFIG.API_LEARN_URL;
  loadArticlesFromUrl(url, pageSize);
}

function loadConfig() {
  $.getJSON('/config/config.json', function(data) {
    CONFIG = data;
    loadNews(30)
  });
}

function man()
{
  $("#searchbar").val('');
  var text = "";// = "<ul>";
  for(var i =0; i < CONFIG.cmds.length; i++)
  {
    text += " - " + CONFIG.cmds[i].name + ": " + CONFIG.cmds[i].description+"\n";
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
    var cmd = query.split(" ")[0];
    
    switch(cmd)
    {
      case 'learn':
        $("#searchbar").val('');
        learning();
        break;

      case 'man':
        $("#searchbar").val('');
        man();
        break;
      
      case 'more':
        $("#searchbar").val('');
        more();
        break;
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